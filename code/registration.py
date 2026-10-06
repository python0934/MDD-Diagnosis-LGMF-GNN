"""registration.py - 配准（头动校正用 scipy，结构配准/归一化用 dipy）。

    * 头动校正：同模态 BOLD->BOLD，刚体(6参数) + 归一化互相关(NCC)，scipy Powell 实现（快，已够用）；
    * 结构配准 BOLD->T1：刚体(6参数) + 互信息(MI)，dipy 实现（稳健）；
    * 空间归一化 T1->MNI：仿射(12参数) + 互信息(MI)，dipy 实现（稳健）。

dipy 的配准是 C 优化 + 解析梯度，远比我手写的 scipy Powell 稳健（尤其跨模态 MI）。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import minimize

import utils


# =====================================================================
# 变换矩阵构造（scipy 头动校正部分用）
# =====================================================================
def rotation_matrix(rx, ry, rz):
    """世界坐标旋转矩阵（弧度）。旋转顺序：R = Rz @ Ry @ Rx。"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def rigid_matrix(params):
    """params = [tx,ty,tz, rx,ry,rz] -> 4x4 世界变换（dst = M @ src）。"""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = rotation_matrix(params[3], params[4], params[5])
    M[:3, 3] = params[:3]
    return M


def _matrix_from_params(p, dof=6):
    if dof == 6:
        return rigid_matrix(p)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = np.asarray(p[0:9]).reshape(3, 3) + np.eye(3)
    M[:3, 3] = p[9:12]
    return M


# =====================================================================
# scipy NCC（头动校正）
# =====================================================================
def _mask_and_values(fixed):
    pos = fixed[fixed > 0]
    thr = float(np.percentile(pos, 10)) if pos.size >= 100 else 0.0
    mask = fixed > thr
    return mask, fixed[mask].ravel().astype(np.float64)


def _neg_ncc(f_masked, m_masked):
    if f_masked.size < 100:
        return 1e9
    f = f_masked - f_masked.mean()
    m = m_masked - m_masked.mean()
    denom = np.sqrt((f * f).sum() * (m * m).sum())
    if denom == 0:
        return 1e9
    return -(f * m).sum() / denom


def register(moving, moving_affine, fixed, fixed_affine, fixed_shape,
             init, dof=6, cost="ncc", pyramid=(2, 1), iters=(30, 10)):
    """scipy Powell 刚体/仿射配准（仅用于同模态 NCC 头动校正）。"""
    init = np.asarray(init, dtype=np.float64)
    best = init.copy()
    for scale, it in zip(pyramid, iters):
        if scale != 1:
            zf = 1.0 / scale
            fz = zoom(fixed, zf, order=1)
            mz = zoom(moving, zf, order=1)
            fa = fixed_affine.copy()
            ma = moving_affine.copy()
            fa[:3, :3] = fa[:3, :3] * scale
            ma[:3, :3] = ma[:3, :3] * scale
            fs = fz.shape
        else:
            fz, mz, fa, ma = fixed, moving, fixed_affine, moving_affine
            fs = fixed_shape
        mask, f_masked = _mask_and_values(fz)

        def objective(p):
            M = _matrix_from_params(p, dof)
            moved = utils.resample_with_affine(mz, ma, M, fa, fs, order=1)
            return _neg_ncc(f_masked, moved[mask].ravel().astype(np.float64))

        res = minimize(objective, best, method="Powell",
                       options={"maxiter": it, "xtol": 1e-3, "ftol": 1e-4})
        best = res.x
    return _matrix_from_params(best, dof), best


# =====================================================================
# dipy MI（结构配准 + 空间归一化）
# =====================================================================
def _dipy_register(moving, moving_affine, fixed, fixed_affine, dof):
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
    from dipy.align.transforms import RigidTransform3D, AffineTransform3D

    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.3)
    affreg = AffineRegistration(metric=metric, level_iters=[3000, 800, 150],
                                sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1])
    if dof == "rigid":
        t = RigidTransform3D()
        t = affreg.optimize(fixed, moving, t, params0=None,
                            static_grid2world=fixed_affine, moving_grid2world=moving_affine)
    else:
        r = RigidTransform3D()
        r = affreg.optimize(fixed, moving, r, params0=None,
                            static_grid2world=fixed_affine, moving_grid2world=moving_affine)
        t = AffineTransform3D()
        t = affreg.optimize(fixed, moving, t, params0=None,
                            static_grid2world=fixed_affine, moving_grid2world=moving_affine,
                            starting_affine=r.affine)
    return t.affine


# =====================================================================
# 对外接口
# =====================================================================
def rigid_align(moving, moving_affine, fixed, fixed_affine, fixed_shape,
                init=(0, 0, 0, 0, 0, 0), cost="ncc", iters=(30, 10), method=None):
    """刚体配准。cost="ncc" 用 scipy（头动校正）；cost="mi" 用 dipy（结构配准）。"""
    if cost == "ncc":
        return register(moving, moving_affine, fixed, fixed_affine, fixed_shape,
                        init=np.asarray(init, dtype=np.float64), dof=6, cost="ncc",
                        iters=iters)
    M = _dipy_register(moving, moving_affine, fixed, fixed_affine, dof="rigid")
    return M, np.zeros(6)


def affine_align(moving, moving_affine, fixed, fixed_affine, fixed_shape, cost="mi"):
    """仿射配准（dipy，空间归一化用）。"""
    M = _dipy_register(moving, moving_affine, fixed, fixed_affine, dof="affine")
    return M, np.zeros(12)