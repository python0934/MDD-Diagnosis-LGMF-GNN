"""ants_registration.py - 基于 ANTs(antspyx) 的配准。

依赖：pip install antspyx

    * 结构配准 BOLD->T1：刚体(Rigid)；
    * 空间归一化 T1->MNI：刚体+仿射+SyN（SyNRA）；
    * 应用变换：把 4D BOLD 逐卷变换到 MNI 空间。

nibabel 与 ants 之间的转换：antspyx 0.5+ 去掉了 from_nibabel/to_nibabel，
改用 from_numpy + 手工从 affine 拆出 origin/spacing/direction。
"""
from __future__ import annotations

import numpy as np
import nibabel as nib


def _nib_to_ants(img):
    """nibabel 图像 -> ants 图像（从 affine 拆 origin/spacing/direction）。"""
    import ants
    data = np.asarray(img.dataobj, dtype=np.float32).astype(np.float32)
    aff = img.affine.astype(np.float64)
    spacing = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))          # 各列模长 = 体素尺寸(正)
    direction = aff[:3, :3] / spacing                           # 方向余弦矩阵
    origin = aff[:3, 3]
    return ants.from_numpy(data, origin=origin.tolist(),
                           spacing=spacing.tolist(),
                           direction=direction.tolist())


def _ants_to_nib(img):
    """ants 图像 -> nibabel 图像（由 spacing/direction/origin 重建 affine）。"""
    data = np.asarray(img.numpy(), dtype=np.float32)
    spacing = np.array(img.spacing, dtype=np.float64)
    direction = np.array(img.direction, dtype=np.float64).reshape(3, 3)
    origin = np.array(img.origin, dtype=np.float64)
    aff = np.eye(4)
    aff[:3, :3] = direction @ np.diag(spacing)
    aff[:3, 3] = origin
    return nib.Nifti1Image(data, aff)


def brain_extraction(t1_nib):
    """ANTs 脑提取（N4 偏场校正 + Atropos 三组织分割）。返回 (mask_nib, brain_t1_nib)。"""
    import ants
    img = _nib_to_ants(t1_nib)
    # 1) N4 偏场校正
    img_n4 = ants.n4_bias_field_correction(img)
    # 2) Otsu 前景掩膜（排除空气）
    head = ants.get_mask(img_n4, cleanup=2)
    # 3) Atropos 三组织分割（CSF/GM/WM）
    seg = ants.atropos(a=img_n4, x=head, i="Kmeans[3]", m="[0.3,1x1]", c="[3,0]")
    seg_img = seg["segmentation"]
    # 4) 脑掩膜 = 非背景组织，取最大连通域去除孤立噪声
    brain = seg_img > 0
    brain = ants.iMath(brain, "GetLargestComponent")
    brain_t1 = ants.mask_image(img_n4, brain)
    return _ants_to_nib(brain), _ants_to_nib(brain_t1)

def coregister_bold_to_t1(mean_bold_nib, t1_nib, outprefix=None):
    """BOLD(平均)->T1 刚体配准。返回 fwdtransforms（映射 moving->fixed）。"""
    import ants
    fixed = _nib_to_ants(t1_nib)
    moving = _nib_to_ants(mean_bold_nib)
    reg = ants.registration(fixed=fixed, moving=moving,
                            type_of_transform="Rigid",
                            outprefix=str(outprefix) if outprefix else "")
    return reg["fwdtransforms"]


def normalize_t1_to_mni(t1_nib, mni_t1_nib, mode="syn", outprefix=None):
    """T1->MNI 配准。mode 支持 rigid / affine / syn。"""
    import ants
    transform_types = {"rigid": "Rigid", "affine": "Affine", "syn": "SyNRA"}
    mode = str(mode).lower()
    if mode not in transform_types:
        raise ValueError(f"不支持的 normalize_dof={mode!r}；应为 rigid/affine/syn")
    fixed = _nib_to_ants(mni_t1_nib)
    moving = _nib_to_ants(t1_nib)
    reg = ants.registration(fixed=fixed, moving=moving,
                            type_of_transform=transform_types[mode],
                            outprefix=str(outprefix) if outprefix else "")
    warped_nib = _ants_to_nib(reg["warpedmovout"])
    return reg["fwdtransforms"], warped_nib


def apply_transforms_to_4d(bold4d_nib, mni_t1_nib, transformlist):
    """把 4D BOLD 逐卷变换到 MNI 空间。返回 (warped_4d_data, affine)。"""
    import ants
    fixed = _nib_to_ants(mni_t1_nib)
    data = np.asarray(bold4d_nib.dataobj, dtype=np.float32)
    affine = bold4d_nib.affine
    T = data.shape[3]
    out = np.zeros((*fixed.shape, T), dtype=np.float32)
    for t in range(T):
        moving = _nib_to_ants(nib.Nifti1Image(data[..., t], affine))
        warped = ants.apply_transforms(fixed=fixed, moving=moving,
                                       transformlist=transformlist,
                                       interpolator="linear")
        out[..., t] = np.asarray(warped.numpy(), dtype=np.float32)
    out_affine = _ants_to_nib(fixed).affine
    return out, out_affine
