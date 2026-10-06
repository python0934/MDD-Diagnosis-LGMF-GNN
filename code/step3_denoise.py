"""step3_denoise.py - 第 7 步 去趋势滤波 + 第 8 步 噪声回归（矩阵/时序计算可用 GPU）。

    7) 对数据与混杂变量执行相同的多项式去趋势 + 频域带通滤波（默认 0.01~0.1 Hz）；
    8) 噪声回归：GLM 回归掉 [头动参数+一阶差分, WM/CSF 平均信号, (可选)全脑平均]；
    最后（可选）每体素 z 标准化。

数值计算（lstsq、FFT）走 utils.xp()：启用 GPU 时用 cupy，否则自动回退 numpy。
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import nibabel as nib
from tqdm import tqdm

import utils


def bandpass(signals, tr, low, high):
    """FFT 带通滤波（矩形窗）。signals: (n_vox, T)，返回同形状。"""
    nyquist = 0.5 / float(tr)
    if not (0 <= float(low) < float(high) <= nyquist):
        raise ValueError(
            f"滤波频率必须满足 0 <= high_pass < low_pass <= Nyquist({nyquist:.4g} Hz)"
        )
    xp_ = utils.xp()
    T = signals.shape[1]
    f = xp_.fft.rfftfreq(T, d=tr)
    spec = xp_.fft.rfft(signals, axis=1)
    mask = ((f >= low) & (f <= high)).astype(spec.dtype)
    spec = spec * mask[None, :]
    return xp_.fft.irfft(spec, n=T, axis=1).astype(signals.dtype)


def run(paths, cfg, ctx, meta_row=None):
    """去噪主流程。返回更新后的 SubjPaths。"""
    paths.restore_existing()
    subj_id = paths.subj_id
    tqdm.write(f"[{subj_id}] 开始去噪")
    paths.require("bold_smooth")

    tr = ctx["tr_map"].get(int(meta_row["protocol"]), cfg.tr_default)

    img = nib.load(str(paths.bold_smooth))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)   # (x, y, z, T)
    affine = img.affine

    # WM/CSF 混杂信号应从未平滑数据提取，避免灰质信号被平滑进组织掩膜。
    confound_data = data
    if paths.bold_norm and Path(paths.bold_norm).exists():
        confound_data = np.asarray(
            nib.load(str(paths.bold_norm)).get_fdata(dtype=np.float32), dtype=np.float32
        )
        if confound_data.shape != data.shape:
            raise ValueError(f"[{subj_id}] norm 与 smooth 数据维度不一致")

    # ---- 脑掩膜：MNI 模板掩膜 且 图像有信号 ----
    mni_mask = utils.resample_with_affine(
        (np.asarray(ctx["mni"]["mask"].get_fdata(dtype=np.float32), dtype=np.float32) > 0.5),
        ctx["mni"]["mask"].affine, np.eye(4), affine, data.shape[:3], order=0,
    ) > 0.5
    brain = mni_mask & (data.mean(axis=3) > 0)
    Y = data[brain].astype(np.float32)            # (n_vox, T)
    if Y.shape[0] == 0:
        raise ValueError(f"[{subj_id}] 脑掩膜内没有有效体素")
    T = Y.shape[1]
    if cfg.detrend_degree < 0 or cfg.detrend_degree >= T:
        raise ValueError("detrend_degree 必须大于等于 0 且小于时间点数")

    # ---- 构建趋势项与噪声项 ----
    t = np.linspace(-1, 1, T)
    trend_cols = [np.ones(T, dtype=np.float32)]
    for d in range(1, cfg.detrend_degree + 1):    # 多项式趋势
        trend_cols.append((t ** d).astype(np.float32))
    trend = np.column_stack(trend_cols).astype(np.float32)
    nuisance_cols = []

    if ("motion" in cfg.nuisance and cfg.do_motion
            and paths.mc_params and Path(paths.mc_params).exists()):
        mc = np.loadtxt(paths.mc_params, skiprows=1).astype(np.float32)      # (T, 6) 跳过表头
        mc = np.atleast_2d(mc)
        if mc.shape[0] != T:
            raise ValueError(f"[{subj_id}] 头动参数长度 {mc.shape[0]} 与 BOLD 时间点数 {T} 不符")
        nuisance_cols.append(mc)                                # 6 个运动参数
        nuisance_cols.append(np.vstack([
            np.zeros((1, mc.shape[1]), dtype=np.float32), np.diff(mc, axis=0)
        ]))                                                     # 一阶差分（速度）
    elif "motion" in cfg.nuisance and cfg.do_motion:
        logging.warning("[%s] 缺少头动参数，未加入 motion 回归量", subj_id)

    for key in ("wm", "csf"):
        if key in cfg.nuisance:
            mp = getattr(paths, f"{key}_mask")
            if mp and Path(mp).exists():
                m = np.asarray(nib.load(str(mp)).get_fdata(dtype=np.float32), dtype=np.float32) > 0.5
                if m.shape != data.shape[:3]:
                    raise ValueError(f"[{subj_id}] {key.upper()} 掩膜与 BOLD 网格不一致")
                if not np.any(m):
                    raise ValueError(f"[{subj_id}] {key.upper()} 掩膜为空")
                nuisance_cols.append(
                    confound_data[m].mean(axis=0).astype(np.float32)[:, None]
                )
            else:
                raise FileNotFoundError(f"[{subj_id}] 缺少 {key.upper()} 掩膜，无法按配置去噪")

    if cfg.include_global:
        nuisance_cols.append(confound_data[brain].mean(axis=0).astype(np.float32)[:, None])

    if cfg.scrub:
        paths.require("mc_fd")
        fd = np.atleast_1d(np.loadtxt(paths.mc_fd, skiprows=1)).astype(np.float32)
        if fd.shape[0] != T:
            raise ValueError(f"[{subj_id}] FD 长度 {fd.shape[0]} 与 BOLD 时间点数 {T} 不符")
        bad = np.flatnonzero(fd > cfg.fd_thr)
        if bad.size:
            nuisance_cols.append(np.eye(T, dtype=np.float32)[:, bad])
            logging.info("[%s] 为 %d/%d 个高 FD 时间点加入 spike 回归量", subj_id, bad.size, T)

    # ---- GPU/CPU：趋势投影 -> 同步滤波数据和混杂变量 -> GLM ----
    xp_ = utils.xp()
    Yd = xp_.asarray(Y)
    D = xp_.asarray(trend)
    trend_beta = xp_.linalg.lstsq(D, Yd.T, rcond=None)[0]
    filtered_y = Yd - (D @ trend_beta).T
    filtered_y = bandpass(filtered_y, tr, cfg.high_pass, cfg.low_pass)

    resid = filtered_y
    if nuisance_cols:
        nuisance = np.column_stack(nuisance_cols).astype(np.float32)
        if nuisance.shape[1] + trend.shape[1] >= T:
            raise ValueError(
                f"[{subj_id}] 混杂变量过多（{nuisance.shape[1]} 列，{T} 个时间点），"
                "没有足够残余自由度"
            )
        N = xp_.asarray(nuisance)
        nuisance_trend_beta = xp_.linalg.lstsq(D, N, rcond=None)[0]
        N = N - D @ nuisance_trend_beta
        N = bandpass(N.T, tr, cfg.high_pass, cfg.low_pass).T
        beta = xp_.linalg.lstsq(N, filtered_y.T, rcond=None)[0]
        resid = filtered_y - (N @ beta).T

    if cfg.standardize:
        resid = (resid - xp_.mean(resid, axis=1, keepdims=True)) / (
            xp_.std(resid, axis=1, keepdims=True) + 1e-8
        )

    clean = utils.to_numpy(resid).astype(np.float32)

    # ---- 写回体空间 ----
    out = np.zeros((data.shape[0], data.shape[1], data.shape[2], T), dtype=np.float32)
    out[brain] = clean
    paths.bold_clean = utils.save_nii(out, affine,
                                       paths.out_dir / "func" / f"{subj_id}_clean.nii.gz",
                                       tr=tr)
    return paths
