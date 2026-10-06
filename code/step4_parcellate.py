"""step4_parcellate.py - AAL116 脑区配准 + 体素均值 + 脑区时序矩阵。

流程：
    1) 把 AAL116 标签图重采样（最近邻）到去噪后 BOLD 的网格（两者都在 MNI 空间，直接对齐）；
    2) 对每个脑区，求该区所有体素时间序列的平均 -> 得到 (N_region, T) 脑区时序矩阵；
    3) 输出 .npy / .csv 矩阵 + 标签 + BOLD 空间标签图。

体素均值计算用 one-hot 矩阵在 GPU（cupy，若有）上向量化完成。
"""
from __future__ import annotations

import sys
import csv
import logging
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import nibabel as nib
from tqdm import tqdm

import utils


def run(paths, cfg, ctx, meta_row=None):
    """脑区时序矩阵提取。返回更新后的 SubjPaths。"""
    paths.restore_existing()
    subj_id = paths.subj_id
    tqdm.write(f"[{subj_id}] 开始 AAL 脑区时序提取")
    paths.require("bold_clean")

    img = nib.load(str(paths.bold_clean))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)   # (x, y, z, T)
    affine = img.affine

    # ---- 1) AAL 图谱重采样到 BOLD 网格（最近邻）----
    aal_img, regions = ctx["aal"]            # regions: [(脑区ID, 名称), ...]
    aal_data = np.asarray(aal_img.dataobj).astype(np.int16)   # 整数标签图用 dataobj 读取
    lab = utils.resample_with_affine(aal_data, aal_img.affine, np.eye(4),
                                     affine, data.shape[:3], order=0).astype(np.int16)
    paths.aal = utils.save_nii(
        lab, affine, paths.out_dir / "parcellation" / f"{subj_id}_aal116.nii.gz",
        dtype=np.int16,
    )

    n_regions = len(regions)
    if n_regions == 0:
        raise ValueError("AAL 图谱没有可用脑区")

    # ---- 2) 体素均值：只统计去噪脑掩膜内的有效体素 ----
    # clean 图像在脑外严格为 0；若把这些 0 也计入分母，会人为压低靠近脑边界的脑区信号。
    valid = np.any(np.isfinite(data) & (data != 0), axis=3)
    if not np.any(valid):
        raise ValueError(f"[{subj_id}] clean 图像内没有有效体素")
    Y = data[valid].astype(np.float32)                             # (n_valid, T)
    lab_valid = lab[valid]
    region_ids = np.asarray([vid for vid, _ in regions], dtype=np.int16)
    R = (lab_valid[:, None] == region_ids[None, :]).astype(np.float32)
    counts_np = R.sum(axis=0)
    missing = np.flatnonzero(counts_np == 0)
    if missing.size:
        logging.warning("[%s] %d 个脑区在有效脑掩膜内没有体素，将输出 NaN", subj_id, missing.size)

    xp_ = utils.xp()
    Yd = xp_.asarray(Y)
    Rd = xp_.asarray(R)
    sums = Rd.T @ Yd                                             # (n_regions, T)
    counts = xp_.maximum(Rd.sum(axis=0, keepdims=True).T, 1.0)  # (n_regions, 1)
    ts = utils.to_numpy(sums / counts).astype(np.float32)        # (n_regions, T) 脑区时序
    if missing.size:
        ts[missing] = np.nan

    # ---- 3) 输出矩阵 ----
    paths.ts = paths.out_dir / "parcellation" / f"{subj_id}_aal116_ts.npy"
    np.save(paths.ts, ts)

    csv_path = paths.out_dir / "parcellation" / f"{subj_id}_aal116_ts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["region"] + [f"t{t + 1}" for t in range(ts.shape[1])])
        for r in range(n_regions):
            name = regions[r][1]
            w.writerow([name] + [f"{v:.6f}" for v in ts[r]])

    with open(paths.out_dir / "parcellation" / f"{subj_id}_aal116_labels.csv",
              "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label"])
        for vid, name in regions:
            w.writerow([vid, name])

    return paths
