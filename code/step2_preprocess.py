"""step2_preprocess.py - 预处理（对应你流程的第 2 步，含 6 个子步骤）。

    1) 时间步长（切片时间）校正   slice_timing_correction
    2) 头动校正（刚体配准）       motion_correction
    3) 结构配准 BOLD -> T1        registration.rigid_align（互信息）
    4) 脑区分割（GM/WM/CSF）      MNI 组织概率图映射（纯 Python 近似 FAST/SPM）
    5) 空间归一化 T1 -> MNI       registration.affine_align，并应用到 BOLD
    6) 空间平滑（FWHM 高斯核）    spatial_smooth

第 7 步 去趋势滤波、第 8 步 噪声回归 在 step3_denoise.py 完成。
本模块输出头动参数（供第 8 步使用）与 GM/WM/CSF 掩膜（脑区分割产物）。
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import utils
import registration


PREPROCESS_SCHEMA_VERSION = 2


def _file_identity(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _preprocess_signature(paths, cfg, tr, slice_order, mni_t1):
    """描述会影响预处理结果的输入与参数，用于安全断点续跑。"""
    if isinstance(slice_order, str):
        serialized_order = slice_order
    else:
        serialized_order = [int(v) for v in slice_order]
    return {
        "schema": PREPROCESS_SCHEMA_VERSION,
        "bold_4d": _file_identity(paths.bold_4d),
        "t1_brain": _file_identity(paths.t1_brain),
        "tr": float(tr),
        "slice_order": serialized_order,
        "slice_ref": cfg.slice_ref,
        "do_slice_timing": bool(cfg.do_slice_timing),
        "do_motion": bool(cfg.do_motion),
        "motion_ref": int(cfg.motion_ref),
        "do_coreg": bool(cfg.do_coreg),
        "do_normalize": bool(cfg.do_normalize),
        "normalize_dof": str(cfg.normalize_dof),
        "do_smooth": bool(cfg.do_smooth),
        "fwhm_mm": float(cfg.fwhm_mm),
        "mni_shape": [int(v) for v in mni_t1.shape[:3]],
        "mni_affine": np.asarray(mni_t1.affine).round(8).tolist(),
    }


def _read_signature(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, TypeError):
        return None


def _write_signature(path, signature):
    path = Path(path)
    utils.ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(signature, fh, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


# =====================================================================
# 1) 时间步长（切片时间）校正
# =====================================================================
def slice_timing_correction(data4d, tr, slice_order="ascending", ref="middle"):
    """对 4D 数据做切片时间校正。

    原理：一个 volume 内不同层并非同时采集，而是按 slice_order 依次采集；
    用傅里叶平移把每一层的时间序列对齐到参考时间点 ref_t。

    参数
    ----
    data4d      : (x, y, z, t) float32
    tr          : 重复时间（秒）
    slice_order : "ascending" / "descending" / "interleaved" / 或显式层索引列表
    ref         : 参考时间点 "middle"(TR/2) 或层索引
    """
    n_slices = data4d.shape[2]
    T = data4d.shape[3]
    if not np.isfinite(tr) or float(tr) <= 0:
        raise ValueError("TR 必须为正数")

    if slice_order == "ascending":
        acquisition_sequence = np.arange(n_slices)
    elif slice_order == "descending":
        acquisition_sequence = np.arange(n_slices)[::-1]
    elif slice_order == "interleaved":
        acquisition_sequence = np.concatenate(
            [np.arange(0, n_slices, 2), np.arange(1, n_slices, 2)]
        )
    else:
        acquisition_sequence = np.asarray(slice_order, dtype=int)

    if (acquisition_sequence.shape != (n_slices,)
            or set(acquisition_sequence.tolist()) != set(range(n_slices))):
        raise ValueError("slice_order 必须包含每个层索引且仅出现一次")

    # acquisition_sequence 是按时间排列的层索引；反转得到每层的采集次序。
    acquisition_rank = np.empty(n_slices, dtype=int)
    acquisition_rank[acquisition_sequence] = np.arange(n_slices)
    acq_t = acquisition_rank / n_slices * tr
    ref_t = tr / 2.0 if ref == "middle" else acq_t[int(ref)]

    out = data4d.copy()
    freqs = np.fft.fftfreq(T, d=tr)
    for k in tqdm(range(n_slices), desc="slice-timing", leave=False):
        delta = ref_t - acq_t[k]            # 从该层采集时刻前移到参考时刻
        if abs(delta) < 1e-9:
            continue
        # 该层所有体素的时间序列 (n_xy, T)
        ts = data4d[:, :, k, :].reshape(-1, T).astype(np.float32)
        spec = np.fft.fft(ts.astype(np.complex64), axis=1)   # 频域
        shift = np.exp(2j * np.pi * freqs * delta).astype(np.complex64)
        spec = spec * shift[None, :]                         # 相位平移
        out[:, :, k, :] = np.fft.ifft(spec, axis=1).real.astype(np.float32).reshape(
            data4d[:, :, k, :].shape
        )
    return out


# =====================================================================
# 2) 头动校正（刚体配准到参考卷）
# =====================================================================
def motion_correction(data4d, affine, ref_index=0, cost="ncc"):
    """逐时间点把 volume 刚体配准到参考卷，实现头动校正。

    返回 (校正后的 4D 数据, 头动参数矩阵 (T, 6))，参数顺序为 [tx,ty,tz,rx,ry,rz]。
    """
    T = data4d.shape[3]
    if not 0 <= int(ref_index) < T:
        raise ValueError(f"motion_ref={ref_index} 超出有效范围 0..{T - 1}")
    ref = data4d[..., ref_index].astype(np.float32)
    params = np.zeros((T, 6), dtype=np.float32)
    out = np.zeros_like(data4d)
    out[..., ref_index] = ref

    prev = np.zeros(6, dtype=np.float64)
    for t in tqdm(range(T), desc="motion-correction", leave=False):
        if t == ref_index:
            continue
        # 每个时间点作为 moving，配准到参考卷 fixed
        M, p = registration.rigid_align(
            data4d[..., t], affine, ref, affine, ref.shape,
            init=prev, cost=cost,
        )
        params[t] = p
        prev = p
        # 应用刚体变换重采样，使该卷与参考卷对齐
        out[..., t] = utils.resample_with_affine(
            data4d[..., t], affine, M, affine, ref.shape, order=1,
        ).astype(np.float32)
    return out, params


def framewise_displacement(params, radius=50.0):
    """由头动参数计算帧间位移 FD（mm）。旋转角度换算为弧长（radius≈50mm）。"""
    p = params.astype(np.float64).copy()
    p[:, 3:6] = p[:, 3:6] * radius
    d = np.abs(np.diff(p, axis=0))
    fd = np.concatenate([[0.0], d.sum(axis=1)])
    return fd


# =====================================================================
# 5) 空间平滑
# =====================================================================
def spatial_smooth(data4d, affine, fwhm_mm):
    """用各向同性高斯核对 4D 数据做空间平滑（不在时间维平滑）。"""
    vox = np.abs(np.diag(affine)[:3])
    vox = np.clip(vox, 1e-3, None)
    sigma = (fwhm_mm / 2.3548) / vox          # FWHM -> 高斯 sigma（按体素）
    return gaussian_filter(data4d, sigma=[sigma[0], sigma[1], sigma[2], 0.0],
                           mode="nearest").astype(np.float32)


# =====================================================================
# 主流程
# =====================================================================
def run(paths, cfg, ctx, meta_row=None):
    """执行预处理（支持断点续跑：已完成的子步骤自动跳过）。返回更新后的 SubjPaths。"""
    paths.restore_existing()
    subj_id = paths.subj_id
    tqdm.write(f"[{subj_id}] 开始预处理")
    if not (cfg.do_coreg and cfg.do_normalize):
        raise ValueError("当前 MNI 流程要求 do_coreg 和 do_normalize 同时开启")
    paths.require("bold_4d", "t1_brain")
    func = paths.out_dir / "func"
    st_f = func / f"{subj_id}_st.nii.gz"
    mc_f = func / f"{subj_id}_mc.nii.gz"

    protocol = int(meta_row["protocol"])
    tr = ctx["tr_map"].get(protocol, cfg.tr_default)
    slice_order = ctx.get("slice_order_map", {}).get(protocol, cfg.slice_order)
    mni = ctx["mni"]
    mni_t1_nib = mni["t1"]
    signature_f = paths.out_dir / "preprocess_signature.json"
    signature = _preprocess_signature(paths, cfg, tr, slice_order, mni_t1_nib)
    reuse_existing = _read_signature(signature_f) == signature
    if not reuse_existing:
        tqdm.write(f"[{subj_id}] 预处理输入/参数已变化或缺少签名，将重新计算")

    # 整个预处理已完成 → 直接跳过
    smooth_f = func / f"{subj_id}_smooth.nii.gz"
    t1_norm_f = paths.out_dir / "anat" / f"{subj_id}_T1w_norm.nii.gz"
    completed = [paths.bold_smooth, paths.bold_norm, paths.gm_mask,
                 paths.wm_mask, paths.csf_mask, t1_norm_f]
    if cfg.do_motion:
        completed.extend([paths.bold_mc, paths.mc_params, paths.mc_fd])
    if reuse_existing and all(p is not None and Path(p).exists() for p in completed):
        tqdm.write(f"[{subj_id}] 预处理已完成，跳过")
        return paths

    # ---- 1) 切片时间校正 ----
    if cfg.do_slice_timing:
        if reuse_existing and st_f.exists():
            paths.bold_st = st_f
            tqdm.write(f"[{subj_id}] 切片时间校正已完成，跳过")
        else:
            img = nib.load(str(paths.bold_4d))
            data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
            affine = img.affine.astype(np.float64)
            data = slice_timing_correction(data, tr, slice_order, cfg.slice_ref)
            paths.bold_st = utils.save_nii(data, affine, st_f, tr=tr)
        cur_f = st_f
    else:
        cur_f = paths.bold_4d

    # 加载当前 4D 数据（切片校正后 或 原始）
    img = nib.load(str(cur_f))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    affine = img.affine.astype(np.float64)

    # ---- 2) 头动校正 ----
    if cfg.do_motion:
        if (reuse_existing and mc_f.exists()
                and paths.mc_params.exists() and paths.mc_fd.exists()):
            paths.bold_mc = mc_f
            mc_img = nib.load(str(mc_f))
            data = np.asarray(mc_img.get_fdata(dtype=np.float32), dtype=np.float32)
            affine = mc_img.affine.astype(np.float64)
            tqdm.write(f"[{subj_id}] 头动校正已完成，跳过")
        else:
            data, mc_params = motion_correction(data, affine, ref_index=cfg.motion_ref)
            paths.bold_mc = utils.save_nii(data, affine, mc_f, tr=tr)
            np.savetxt(paths.mc_params, mc_params, delimiter="\t",
                       header="tx ty tz rx ry rz", comments="")
            fd = framewise_displacement(mc_params)
            np.savetxt(paths.mc_fd, fd, delimiter="\t", header="fd", comments="")

    # ---- 3) 结构配准 BOLD -> T1（dipy 刚体，互信息）----
    t1_brain_img = nib.load(str(paths.t1_brain))
    t1_affine = t1_brain_img.affine.astype(np.float64)
    mean_bold = data.mean(axis=3).astype(np.float32)

    M_coreg = np.eye(4, dtype=np.float64)
    if cfg.do_coreg:
        tqdm.write(f"[{subj_id}] BOLD->T1 刚体配准（dipy MI）")
        M_coreg, _ = registration.rigid_align(
            mean_bold, affine,
            np.asarray(t1_brain_img.get_fdata(dtype=np.float32), dtype=np.float32),
            t1_affine, t1_brain_img.shape, cost="mi",
        )

    # ---- 5) 空间归一化 T1 -> MNI（dipy 仿射，互信息）----
    M_norm = np.eye(4, dtype=np.float64)
    if cfg.do_normalize:
        tqdm.write(f"[{subj_id}] T1->MNI 仿射配准（dipy MI）")
        M_norm, _ = registration.affine_align(
            np.asarray(t1_brain_img.get_fdata(dtype=np.float32), dtype=np.float32),
            t1_affine,
            np.asarray(mni_t1_nib.get_fdata(dtype=np.float32), dtype=np.float32),
            mni_t1_nib.affine.astype(np.float64), mni_t1_nib.shape,
        )
        # T1 归一化到 MNI（用于质控查看）
        t1_norm_data = utils.resample_with_affine(
            np.asarray(t1_brain_img.get_fdata(dtype=np.float32), dtype=np.float32),
            t1_affine, M_norm, mni_t1_nib.affine.astype(np.float64),
            mni_t1_nib.shape[:3], order=1,
        )
        utils.save_nii(t1_norm_data, mni_t1_nib.affine, t1_norm_f)

    # BOLD 变换到 MNI（组合变换：BOLD_world -> T1_world -> MNI_world）
    mni_affine = mni_t1_nib.affine.astype(np.float64)
    T = data.shape[3]
    norm = np.zeros((*mni_t1_nib.shape[:3], T), dtype=np.float32)
    tqdm.write(f"[{subj_id}] 将 BOLD 变换到 MNI 空间（{T} 个时间点）")
    for t in tqdm(range(T), desc="apply-transforms", leave=False):
        norm[..., t] = utils.resample_with_affine(
            data[..., t], affine, np.linalg.inv(M_norm @ M_coreg),
            mni_affine, mni_t1_nib.shape[:3], order=1,
        ).astype(np.float32)
    paths.bold_norm = utils.save_nii(norm, mni_affine,
                                      paths.out_dir / "func" / f"{subj_id}_norm.nii.gz",
                                      tr=tr)

    # ---- 6) 空间平滑 ----
    if cfg.do_smooth:
        smooth = spatial_smooth(norm, mni["t1"].affine, cfg.fwhm_mm)
    else:
        smooth = norm
    paths.bold_smooth = utils.save_nii(smooth, mni["t1"].affine,
                                        paths.out_dir / "func" / f"{subj_id}_smooth.nii.gz",
                                        tr=tr)

    # ---- 4) 脑区分割：把脑组织分割为 GM / WM / CSF 三类 ----
    # 纯 Python 下用 MNI 组织概率图近似个体组织分割（替代 FSL FAST / SPM）。
    # BOLD 已归一化到 MNI，概率图与 BOLD 同网格，直接阈值(>0.5)得到三类掩膜。
    for key in ("gm", "wm", "csf"):
        prob = np.asarray(mni[key].get_fdata(dtype=np.float32), dtype=np.float32)
        mask = (prob > 0.5).astype(np.float32)
        setattr(paths, f"{key}_mask",
                utils.save_nii(mask, mni["t1"].affine,
                               paths.out_dir / "masks" / f"{subj_id}_{key}_mask.nii.gz"))

    _write_signature(signature_f, signature)
    return paths
