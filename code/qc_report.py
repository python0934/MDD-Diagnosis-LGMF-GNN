"""qc_report.py - 一键 QC 报告（头动指标 + 叠加图 PNG）。

对每个受试者：
    1) 计算头动指标：平均/最大 FD、超阈值(0.5mm)时间点占比、最大平移/旋转；
    2) 生成叠加图：归一化 T1、平均 BOLD(MNI)、WM/CSF 掩膜、AAL116 脑区；
    3) 汇总写入 derivatives/qc/qc_summary.csv，每受试者一个子目录放 PNG。

用法：
    python qc_report.py                        # 全部受试者
    python qc_report.py --subjects 0001 0002   # 指定受试者
    python qc_report.py --subjects 0001 --no-png   # 只看头动指标，不画图
"""
from __future__ import annotations

import sys
import csv
import argparse
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无界面后端，直接写 PNG
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, image
from tqdm import tqdm

import utils
from config import Config


FD_THR = 0.5                   # FD 阈值（mm）


# =====================================================================
# 头动指标
# =====================================================================
def compute_motion(subj_id, func_dir):
    """从 mc_fd / mc_params 计算头动指标，返回 dict。文件缺失则返回 None。"""
    fd_f = func_dir / f"{subj_id}_mc_fd.tsv"
    mc_f = func_dir / f"{subj_id}_mc_params.tsv"
    if not fd_f.exists() or not mc_f.exists():
        return None
    fd = np.loadtxt(fd_f, skiprows=1)
    mc = np.loadtxt(mc_f, skiprows=1)          # (T, 6): tx ty tz rx ry rz
    trans = mc[:, :3]                          # 平移 mm
    rot = np.rad2deg(mc[:, 3:])                # 旋转 度
    trans_mag = np.sqrt((trans ** 2).sum(1))   # 平移向量模长
    return {
        "fd_mean": float(fd.mean()),
        "fd_max": float(fd.max()),
        "fd_over_pct": float((fd > FD_THR).mean() * 100),
        "trans_max_axis_mm": float(np.abs(trans).max()),
        "trans_max_mag_mm": float(trans_mag.max()),
        "rot_max_axis_deg": float(np.abs(rot).max()),
    }


def plot_motion(subj_id, func_dir, out_png):
    """画 FD / 平移 / 旋转三条时间曲线。"""
    fd = np.loadtxt(func_dir / f"{subj_id}_mc_fd.tsv", skiprows=1)
    mc = np.loadtxt(func_dir / f"{subj_id}_mc_params.tsv", skiprows=1)
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(fd, lw=0.6)
    ax[0].axhline(FD_THR, color="r", ls="--", lw=0.8)
    ax[0].set_ylabel("FD (mm)")
    ax[0].set_title(f"{subj_id} head motion (FD)")
    ax[1].plot(mc[:, :3], lw=0.6)
    ax[1].set_ylabel("translation (mm)")
    ax[1].legend(["tx", "ty", "tz"], ncol=3, fontsize=8)
    ax[2].plot(np.rad2deg(mc[:, 3:]), lw=0.6)
    ax[2].set_ylabel("rotation (deg)")
    ax[2].set_xlabel("volume index")
    ax[2].legend(["rx", "ry", "rz"], ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# =====================================================================
# 叠加图
# =====================================================================
def _plot_anat_or_epi(img, title, out_png, is_epi=False):
    """用 nilearn 画 mosaic 切片；mosaic 失败则退回 ortho。"""
    fn = plotting.plot_epi if is_epi else plotting.plot_anat
    try:
        fn(img, display_mode="ortho", title=title, output_file=str(out_png))
    except Exception:
        fn(img, display_mode="ortho", title=title, output_file=str(out_png))


def plot_brain_images(subj_id, paths, out_dir):
    """生成归一化 T1 / 平均 BOLD / WM-CSF 掩膜 / AAL 叠加图。"""
    func = paths.out_dir / "func"
    anat = paths.out_dir / "anat"
    masks = paths.out_dir / "masks"
    parc = paths.out_dir / "parcellation"

    t1_norm_f = anat / f"{subj_id}_T1w_norm.nii.gz"
    norm_f = func / f"{subj_id}_norm.nii.gz"
    wm_f = masks / f"{subj_id}_wm_mask.nii.gz"
    csf_f = masks / f"{subj_id}_csf_mask.nii.gz"
    aal_f = parc / f"{subj_id}_aal116.nii.gz"

    if t1_norm_f.exists():
        _plot_anat_or_epi(image.load_img(str(t1_norm_f)),
                          f"{subj_id} T1w normalized (MNI)",
                          out_dir / f"{subj_id}_T1w_norm.png")

    if norm_f.exists():
        mean_bold = image.mean_img(image.load_img(str(norm_f)))
        _plot_anat_or_epi(mean_bold, f"{subj_id} mean BOLD (MNI)",
                          out_dir / f"{subj_id}_bold_mean.png", is_epi=True)

    # WM / CSF 掩膜叠加在归一化 T1 上
    if t1_norm_f.exists() and wm_f.exists():
        try:
            plotting.plot_roi(str(wm_f), bg_img=str(t1_norm_f), display_mode="ortho",
                              cmap="Reds", title=f"{subj_id} WM mask",
                              output_file=str(out_dir / f"{subj_id}_wm_mask.png"))
        except Exception:
            pass
    if t1_norm_f.exists() and csf_f.exists():
        try:
            plotting.plot_roi(str(csf_f), bg_img=str(t1_norm_f), display_mode="ortho",
                              cmap="Blues", title=f"{subj_id} CSF mask",
                              output_file=str(out_dir / f"{subj_id}_csf_mask.png"))
        except Exception:
            pass

    # AAL116 脑区叠加
    if t1_norm_f.exists() and aal_f.exists():
        try:
            plotting.plot_roi(str(aal_f), bg_img=str(t1_norm_f), display_mode="ortho",
                              cmap="gist_ncar", title=f"{subj_id} AAL116",
                              output_file=str(out_dir / f"{subj_id}_aal116.png"))
        except Exception:
            pass


# =====================================================================
# 主流程
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="SRPBS QC 报告")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--no-png", action="store_true", help="只算头动指标，不画图")
    args = ap.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    if args.out_root:
        cfg.out_root = Path(args.out_root)
    if args.subjects:
        cfg.subjects = args.subjects

    qc_root = utils.ensure_dir(cfg.out_root / "qc")
    subj_dirs = utils.list_subjects(cfg.data_root, cfg.subjects)
    if not subj_dirs:
        print("未发现受试者，请检查 --subjects / --data-root")
        sys.exit(1)

    summary_rows = []
    for sd in tqdm(subj_dirs, desc="QC"):
        subj_id = sd.name
        out_dir = utils.ensure_dir(qc_root / subj_id)
        func = cfg.out_root / subj_id / "func"

        metrics = compute_motion(subj_id, func)
        if metrics is None:
            print(f"[{subj_id}] 缺头动文件，跳过")
            continue

        # 头动曲线
        if not args.no_png:
            plot_motion(subj_id, func, out_dir / f"{subj_id}_motion.png")
            plot_brain_images(subj_id, utils.SubjPaths(subj_id=subj_id, out_dir=cfg.out_root / subj_id),
                              out_dir)

        row = {"subject": subj_id}
        row.update({k: round(v, 4) for k, v in metrics.items()})
        summary_rows.append(row)
        print(f"[{subj_id}] FD均值={metrics['fd_mean']:.3f}mm FD最大={metrics['fd_max']:.3f}mm "
              f"超阈值占比={metrics['fd_over_pct']:.1f}% 最大平移={metrics['trans_max_mag_mm']:.2f}mm "
              f"最大旋转={metrics['rot_max_axis_deg']:.2f}deg")

    # 汇总 CSV
    if summary_rows:
        fields = ["subject", "fd_mean", "fd_max", "fd_over_pct",
                  "trans_max_axis_mm", "trans_max_mag_mm", "rot_max_axis_deg"]
        with open(qc_root / "qc_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print(f"\n汇总已写入：{qc_root / 'qc_summary.csv'}")
    print("QC 完成。PNG 位于：", qc_root)


if __name__ == "__main__":
    main()