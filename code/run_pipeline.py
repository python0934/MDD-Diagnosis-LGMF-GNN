"""run_pipeline.py - SRPBS 预处理总入口。

用法示例：
    python run_pipeline.py                          # 全部受试者，全流程
    python run_pipeline.py --subjects 0001 0002     # 只处理指定受试者
    python run_pipeline.py --start 0 --end 10       # 处理前 10 个（按编号排序）
    python run_pipeline.py --steps merge preprocess # 只跑前两步
    python run_pipeline.py --gpu                    # 启用 GPU（有 cupy 时）
    python run_pipeline.py --gpu off                # 强制 CPU

流程（默认四步，逐步依赖）：
    merge       -> 合并 rsfmri/vol_* 为 4D BOLD，并去颅骨 T1；
    preprocess  -> 切片时间校正 / 头动校正 / 结构配准 / 空间归一化 / 平滑；
    denoise     -> 去趋势 + 带通滤波 + 噪声回归；
    parcellate  -> AAL116 脑区配准 + 体素均值 -> 脑区时序矩阵。
"""
from __future__ import annotations

import sys
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from tqdm import tqdm

import utils
from config import Config


STEP_MODULES = {
    "merge": "step1_merge_4d",
    "preprocess": "step2_preprocess",
    "denoise": "step3_denoise",
    "parcellate": "step4_parcellate",
}


def parse_args():
    ap = argparse.ArgumentParser(description="SRPBS Python/ANTs 预处理流水线")
    ap.add_argument("--data-root", default=None, help="数据根目录（默认 config.py 中值）")
    ap.add_argument("--out-root", default=None, help="输出根目录（默认 config.py 中值）")
    ap.add_argument("--subjects", nargs="*", default=None, help="受试者 id，如 0001 0002")
    ap.add_argument("--start", type=int, default=None, help="起始索引（含）")
    ap.add_argument("--end", type=int, default=None, help="结束索引（不含）")
    ap.add_argument("--steps", nargs="*", default=None,
                    choices=list(STEP_MODULES.keys()), help="要执行的步骤子集")
    ap.add_argument("--gpu", nargs="?", const="on", default=None,
                    choices=("auto", "on", "off"),
                    help="on/off/auto；缺省 auto")
    ap.add_argument("--tr", type=float, default=None, help="默认 TR（秒）")
    ap.add_argument("--fwhm", type=float, default=None, help="平滑核 FWHM（mm）")
    ap.add_argument("--low-pass", type=float, default=None, help="带通滤波上限（Hz）")
    ap.add_argument("--high-pass", type=float, default=None, help="带通滤波下限（Hz）")
    ap.add_argument("--no-slice-timing", action="store_true", help="跳过切片时间校正")
    ap.add_argument("--no-motion", action="store_true", help="跳过头动校正")
    ap.add_argument("--no-smooth", action="store_true", help="跳过空间平滑")
    ap.add_argument("--n-jobs", type=int, default=None, help="并行受试者数（>1 必须 --gpu off）")
    ap.add_argument("--normalize-dof", choices=("rigid", "affine", "syn"), default=None,
                    help="T1 到 MNI 的配准模型")
    ap.add_argument("--scrub", action="store_true", help="为高 FD 时间点加入 spike 回归量")
    ap.add_argument("--fd-thr", type=float, default=None, help="scrub 的 FD 阈值（mm）")
    ap.add_argument("--drop-intermediates", action="store_true",
                    help="成功后删除 st/mc/norm 中间 4D 图像")
    return ap.parse_args()


def decide_gpu(mode):
    """解析 GPU 模式，返回布尔值。"""
    if mode == "off":
        return False
    try:
        import cupy  # noqa: F401
        return True
    except Exception as e:  # noqa: BLE001
        msg = ("cupy 不可用，矩阵/时序计算将使用 CPU(numpy)。"
               "如需 GPU，请安装与本机 CUDA 匹配的 cupy wheel。")
        if mode == "on":
            raise RuntimeError(msg)
        logging.warning(msg)
        return False


def check_deps():
    import numpy, scipy, nibabel, nilearn, dipy, tqdm as _tqdm
    msg = (f"依赖版本 -> numpy {numpy.__version__} | scipy {scipy.__version__} | "
           f"nibabel {nibabel.__version__} | nilearn {nilearn.__version__} | "
           f"dipy {dipy.__version__} | tqdm {_tqdm.__version__}")
    try:
        import ants
        msg += f" | antspyx {ants.__version__}"
    except ImportError:
        msg += " | antspyx 未安装（使用 dipy 配准替代）"
    print(msg)


def build_config(args):
    cfg = Config()
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    if args.out_root:
        cfg.out_root = Path(args.out_root)
    if args.subjects:
        cfg.subjects = args.subjects
    if args.steps:
        cfg.steps = tuple(args.steps)
    if args.gpu is not None:
        cfg.use_gpu = args.gpu
    if args.tr is not None:
        cfg.tr_default = args.tr
    if args.fwhm is not None:
        cfg.fwhm_mm = args.fwhm
    if args.low_pass is not None:
        cfg.low_pass = args.low_pass
    if args.high_pass is not None:
        cfg.high_pass = args.high_pass
    if args.no_slice_timing:
        cfg.do_slice_timing = False
    if args.no_motion:
        cfg.do_motion = False
    if args.no_smooth:
        cfg.do_smooth = False
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs
    if args.normalize_dof is not None:
        cfg.normalize_dof = args.normalize_dof
    if args.scrub:
        cfg.scrub = True
    if args.fd_thr is not None:
        cfg.fd_thr = args.fd_thr
    if args.drop_intermediates:
        cfg.keep_intermediates = False
    return cfg


def validate_config(cfg):
    """在耗时处理开始前拒绝无效或会产生错误空间解释的配置。"""
    if cfg.n_jobs < 1:
        raise ValueError("n_jobs 必须大于等于 1")
    if cfg.tr_default <= 0:
        raise ValueError("tr_default 必须为正数")
    if cfg.fwhm_mm < 0:
        raise ValueError("fwhm_mm 不能为负数")
    if cfg.fd_thr < 0:
        raise ValueError("fd_thr 不能为负数")
    if cfg.normalize_dof not in {"rigid", "affine", "syn"}:
        raise ValueError("normalize_dof 必须是 rigid/affine/syn")
    invalid_nuisance = set(cfg.nuisance) - {"motion", "wm", "csf"}
    if invalid_nuisance:
        raise ValueError(f"未知 nuisance 项：{sorted(invalid_nuisance)}")
    nyquist = 0.5 / cfg.tr_default
    if not (0 <= cfg.high_pass < cfg.low_pass <= nyquist):
        raise ValueError(
            f"滤波频率必须满足 0 <= high_pass < low_pass <= {nyquist:.4g} Hz"
        )
    if "preprocess" in cfg.steps and not (cfg.do_coreg and cfg.do_normalize):
        raise ValueError("当前 MNI 流程要求 do_coreg 和 do_normalize 同时开启")
    if cfg.scrub and not cfg.do_motion:
        raise ValueError("scrub 依赖头动校正和 FD，不能与 do_motion=False 同时使用")


def main():
    args = parse_args()
    cfg = build_config(args)

    try:
        validate_config(cfg)
    except ValueError as e:
        print(f"配置错误：{e}", file=sys.stderr)
        return 2

    utils.ensure_dir(cfg.out_root)
    utils.setup_logging(log_file=cfg.out_root / "pipeline.log")
    try:
        use_gpu = decide_gpu(cfg.use_gpu)
        check_deps()
    except (ImportError, RuntimeError) as e:
        logging.error("运行环境不完整：%s", e)
        return 2
    if use_gpu and cfg.n_jobs > 1:
        logging.error("GPU 模式不支持并行受试者；请使用 --n-jobs 1 或 --gpu off")
        return 2
    utils.set_use_gpu(use_gpu)

    # ---- 元数据与模板（惰性加载）----
    meta = utils.read_participants(cfg.data_root)
    tr_map = utils.get_tr_map(cfg.data_root, cfg.tr_default)
    slice_order_map = utils.get_slice_order_map(cfg.data_root)
    ctx = {"tr_map": tr_map, "slice_order_map": slice_order_map}

    need_mni = bool({"preprocess", "denoise", "parcellate"} & set(cfg.steps))
    need_aal = "parcellate" in cfg.steps
    try:
        if need_mni:
            tqdm.write("加载 MNI 模板 / 组织概率图（首次会在线下载并缓存）...")
            ctx["mni"] = utils.get_mni_templates(cfg)
        if need_aal:
            tqdm.write("加载 AAL116 图谱（首次会在线下载并缓存）...")
            mni_t1 = ctx.get("mni", {}).get("t1") if ctx.get("mni") else None
            ctx["aal"] = utils.get_aal(cfg, mni_t1=mni_t1)
    except Exception as e:  # noqa: BLE001
        logging.error("模板/图谱加载失败：%s", e)
        logging.error("若是 SSL 证书错误或网络受限，请手动下载后，在 config.py 中设置"
                      " mni_t1 / aal_atlas 等本地路径（详见 README「常见问题」）。")
        return 1

    # ---- 受试者列表 ----
    subj_dirs = utils.list_subjects(cfg.data_root, cfg.subjects)
    if args.start is not None or args.end is not None:
        subj_dirs = subj_dirs[args.start:args.end]
    if not subj_dirs:
        logging.error("未发现受试者目录，请检查 --data-root 与 --subjects")
        return 1

    logging.info("共 %d 个受试者待处理；步骤：%s；GPU=%s",
                 len(subj_dirs), ", ".join(cfg.steps), utils.xp().__name__)

    # ---- 逐受试者处理（n_jobs>1 时使用线程；底层 ANTs/numpy 会释放 GIL）----
    def process_subject(sd):
        subj_id = sd.name
        meta_row = meta.get(subj_id)
        if meta_row is None:
            logging.error("[%s] 在 participants.tsv 中无记录", subj_id)
            return False
        try:
            paths = utils.SubjPaths(
                subj_id=subj_id, out_dir=cfg.out_root / subj_id, src_dir=sd
            ).restore_existing()
            for step in cfg.steps:
                mod = __import__(STEP_MODULES[step])
                paths = mod.run(paths, cfg, ctx, meta_row)
            final_output = paths.ts or paths.bold_clean or paths.bold_smooth or paths.bold_4d
            if not cfg.keep_intermediates:
                removed = paths.remove_intermediate_images()
                logging.info("[%s] 已删除 %d 个中间 4D 文件", subj_id, len(removed))
            logging.info("[%s] 完成 -> %s", subj_id, final_output)
            return True
        except Exception as e:  # noqa: BLE001
            logging.error("[%s] 失败：%s", subj_id, e, exc_info=True)
            return False

    if cfg.n_jobs == 1:
        results = [process_subject(sd) for sd in tqdm(subj_dirs, desc="subjects")]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=cfg.n_jobs) as pool:
            futures = [pool.submit(process_subject, sd) for sd in subj_dirs]
            with tqdm(total=len(futures), desc="subjects") as progress:
                for future in as_completed(futures):
                    results.append(future.result())
                    progress.update(1)

    ok = sum(results)
    fail = len(results) - ok
    logging.info("全部结束：成功 %d，失败 %d", ok, fail)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
