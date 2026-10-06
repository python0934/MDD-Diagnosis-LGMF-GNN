"""step1_merge_4d.py - 把 SRPBS 逐时间点 3D NIfTI 合并为 4D BOLD，并准备结构像。

对应流程第 1 步的产物（原始数据 -> 4D NIfTI）：
    * 把 rsfmri/vol_XXX（每个文件是 1 个时间点的 3D EPI）合并为单个 4D NIfTI；
    * 转存 T1 结构像；
    * 对 T1 去颅骨（dipy median_otsu）并生成脑掩膜，供后续配准/归一化使用。
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import nibabel as nib

import utils


def run(paths, cfg, ctx=None, meta_row=None):
    """合并 4D BOLD + 准备 T1（已存在则跳过）。返回更新后的 SubjPaths。"""
    paths.restore_existing()
    subj_id = paths.subj_id
    tr = cfg.tr_default
    if ctx is not None and meta_row is not None:
        tr = ctx.get("tr_map", {}).get(int(meta_row["protocol"]), cfg.tr_default)
    bold4d = paths.out_dir / "func" / f"{subj_id}_bold.nii.gz"
    t1_brain = paths.out_dir / "anat" / f"{subj_id}_T1w_brain.nii.gz"
    required = (paths.bold_4d, paths.t1, paths.t1_brain, paths.brain_mask)
    if all(p is not None and Path(p).exists() for p in required):
        existing = nib.load(str(paths.bold_4d))
        if existing.ndim != 4:
            raise ValueError(f"[{subj_id}] 已有 bold 文件不是 4D NIfTI")
        if not np.isclose(existing.header.get_zooms()[3], tr):
            # 旧版本未写 TR；保留数据内容，仅重写带正确时间元数据的 NIfTI。
            paths.bold_4d = utils.save_nii(
                existing.get_fdata(dtype=np.float32), existing.affine, bold4d, tr=tr
            )
        from tqdm import tqdm
        tqdm.write(f"[{subj_id}] merge 已完成，跳过")
        return paths

    subj_dir = Path(paths.src_dir)
    rsfmri = subj_dir / "rsfmri"
    vols = sorted(rsfmri.glob("vol_*"), key=utils.sort_vol_key)
    if not vols:
        raise FileNotFoundError(f"{subj_dir} 缺少 rsfmri/vol_* 文件")

    # ---- 1) 合并逐时间点 3D NIfTI 为 4D BOLD ----
    img4d = utils.merge_volumes_4d(vols)
    paths.bold_4d = utils.save_nii(
        img4d.get_fdata(dtype=np.float32), img4d.affine,
        paths.out_dir / "func" / f"{paths.subj_id}_bold.nii.gz",
        tr=tr,
    )

    # ---- 2) 转存 T1 结构像 ----
    t1_src = subj_dir / "t1" / "defaced_mprage.nii"
    if not t1_src.exists():
        raise FileNotFoundError(f"缺少 T1 结构像：{t1_src}")
    t1_img = nib.load(str(t1_src))
    paths.t1 = utils.save_nii(
        t1_img.get_fdata(dtype=np.float32), t1_img.affine,
        paths.out_dir / "anat" / f"{paths.subj_id}_T1w.nii.gz",
    )

    # ---- 3) 去颅骨 + 脑掩膜 ----
    t1_data = np.asarray(t1_img.get_fdata(dtype=np.float32), dtype=np.float32)
    masked, mask = utils.skull_strip(t1_data)
    paths.t1_brain = utils.save_nii(
        masked, t1_img.affine, paths.out_dir / "anat" / f"{paths.subj_id}_T1w_brain.nii.gz",
    )
    paths.brain_mask = utils.save_nii(
        mask.astype(np.float32), t1_img.affine, paths.out_dir / "masks" / f"{paths.subj_id}_brain_mask.nii.gz",
    )

    return paths
