"""config.py - 全局配置。

所有可调参数集中在此。run_pipeline.py 会读取默认值，并允许用命令行参数覆盖
部分常用项（TR、FWHM、频段、是否用 GPU、受试者范围等）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Config:
    # ---------------- 路径 ----------------
    # SRPBS 解压后的内层目录（含 data/ 子目录与 participants.tsv 等元数据）
    data_root: Path = Path(r"e:\多模态情感障碍毕设课题\SRPBS_OPEN")
    # 派生数据（预处理结果）输出根目录，每个受试者一个子目录
    out_root: Path = Path(r"e:\多模态情感障碍毕设课题\derivatives")
    # 受试者过滤：空列表 = 全部；否则给出 sub id（如 ["0001", "0002"]）
    subjects: list = field(default_factory=list)

    # ---------------- 运行控制 ----------------
    # 要执行的步骤子集；默认全流程。可选：merge / preprocess / denoise / parcellate
    steps: tuple = ("merge", "preprocess", "denoise", "parcellate")
    n_jobs: int = 1                       # 并行受试者数（>1 时必须关闭 GPU）
    use_gpu: str = "auto"                 # auto | on | off  （auto=有 cupy 就用，否则回退 CPU）
    keep_intermediates: bool = True       # 是否保留中间 4D 产物（st/mc/norm）

    # ---------------- 采集参数 ----------------
    tr_default: float = 2.5               # 默认 TR（秒）；会按 protocol 从 MRI_protocols_rsMRI.tsv 自动覆盖
    slice_order: str = "ascending"        # 协议表无法解析时的后备层序
    slice_ref: str = "middle"             # 切片时间校正参考点：middle（TR/2）或 0 / 层索引

    # ---------------- 预处理开关与参数 ----------------
    do_slice_timing: bool = True          # 1) 时间步长（切片时间）校正
    do_motion: bool = True                # 2) 头动校正（刚体配准）
    motion_ref: int = 0                   # 头动校正的参考时间点（默认第 0 卷）
    do_coreg: bool = True                 # 3) BOLD → T1 结构配准
    do_normalize: bool = True             # 4) 空间归一化（T1 → MNI，并应用到 BOLD）
    normalize_dof: str = "syn"            # 归一化模型：syn（非线性）/ affine / rigid
    do_smooth: bool = True                # 5) 空间平滑
    fwhm_mm: float = 6.0                  # 平滑核 FWHM（毫米）
    mni_resample_mm: float = 3.0          # 归一化后 MNI 网格体素大小（毫米，缩小内存/显存）

    # ---------------- 去噪（去趋势 + 滤波 + 噪声回归）----------------
    detrend_degree: int = 1               # 多项式去趋势阶数：1=线性；2=二次
    low_pass: float = 0.1                 # 带通滤波上限（Hz）
    high_pass: float = 0.01               # 带通滤波下限（Hz）
    nuisance: tuple = ("motion", "wm", "csf")  # 回归的噪声项：motion / wm / csf
    include_global: bool = False          # 是否回归全脑平均信号（默认关，避免引入负相关偏置）
    standardize: bool = True              # 去噪后是否对每体素时间序列 z 标准化（常用 zscore）
    scrub: bool = False                   # 是否为高 FD 时间点加入 spike 回归量（保留时间轴长度）
    fd_thr: float = 0.5                   # 头动剔除阈值 FD（mm）

    # ---------------- 模板与图谱（留空则用 nilearn 在线下载并缓存）----------------
    mni_t1: Path = None                   # 本地 MNI T1 模板路径（可选）
    mni_gm: Path = None                   # 本地 GM 概率图路径（可选）
    mni_wm: Path = None                   # 本地 WM 概率图路径（可选）
    mni_csf: Path = None                  # 本地 CSF 概率图路径（可选）
    mni_mask: Path = None                 # 本地脑掩膜路径（可选）
    aal_atlas: Path = None                # AAL 图谱 NIfTI（脑区体数据）；留空则自动下载 SPM12 版（含正确脑区 ID 映射）
    aal_labels: Path = None               # 本地 AAL 标签文件（可选；留空用自动下载的标签。注意：标准 AAL.nii 脑区 ID 是 2001+，不是 1..116）
    nilearn_data_dir: Path = None         # nilearn 下载缓存目录（None=out_root/nilearn_data）

    def asdict(self):
        return asdict(self)
