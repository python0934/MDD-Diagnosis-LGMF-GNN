"""utils.py - 通用工具。

包含：GPU/CPU 数值后端切换、NIfTI 读写、坐标重采样（应用线性变换）、
受试者发现、协议 TR 解析、MNI 模板 / AAL 图谱加载与重采样、去颅骨等。
"""
from __future__ import annotations

import logging
import re
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform


# =====================================================================
# GPU / CPU 数值后端
# =====================================================================
_USE_GPU = False


def set_use_gpu(flag: bool) -> None:
    """全局设置是否使用 GPU（cupy）。"""
    global _USE_GPU
    _USE_GPU = bool(flag)


def xp():
    """返回当前数值后端：启用且可用时返回 cupy，否则返回 numpy。"""
    if _USE_GPU:
        try:
            import cupy
            return cupy
        except Exception as e:  # noqa: BLE001
            logging.warning("GPU 后端不可用，回退 numpy（CPU）：%s", e)
    return np


def to_numpy(arr):
    """把可能是 cupy 的数组转回 numpy（nibabel 写出需要 numpy 数组）。"""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


# =====================================================================
# 目录 / 日志
# =====================================================================
def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(log_file=None, level=logging.INFO):
    handlers = [logging.StreamHandler()]
    if log_file:
        ensure_dir(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


# =====================================================================
# NIfTI 读写 / 坐标重采样
# =====================================================================
def load_nii(path, dtype=np.float32):
    """读取 NIfTI，返回 (image, data)。"""
    img = nib.load(str(path))
    return img, np.asarray(img.get_fdata(dtype=dtype), dtype=dtype)


def save_nii(data, affine, path, dtype=np.float32, tr=None):
    """把体数据写为 NIfTI（.nii.gz 由后缀自动压缩）。

    4D 数据必须通过 ``tr`` 指定时间分辨率；这样输出文件不会使用
    nibabel 默认的 1 秒时间步长。
    """
    path = Path(path)
    ensure_dir(path.parent)
    arr = np.asarray(data, dtype=dtype)
    img = nib.Nifti1Image(arr, np.asarray(affine, dtype=np.float64))
    img.header.set_xyzt_units("mm", "sec" if arr.ndim == 4 else "unknown")
    if arr.ndim == 4:
        if tr is None or not np.isfinite(tr) or float(tr) <= 0:
            raise ValueError("保存 4D NIfTI 时必须提供正数 TR（秒）")
        img.header.set_zooms((*img.header.get_zooms()[:3], float(tr)))
    nib.save(img, str(path))
    return path


def resample_with_affine(data, src_affine, warp, target_affine, target_shape,
                         order=1, cval=0.0):
    """把源体数据重采样到目标网格，并施加一个线性变换 warp。

    参数
    ----
    data           : 源体数据（3D）
    src_affine     : 源图像 voxel->world 仿射矩阵
    warp           : 4x4 世界坐标变换，含义为 dst_world = warp @ src_world
    target_affine  : 目标网格 voxel->world 仿射矩阵
    target_shape   : 目标网格尺寸（3 元组）
    order          : 插值阶数 0=最近邻 / 1=线性 / 3=三次

    返回
    ----
    重采样后的数据，shape == target_shape。
    """
    # 目标体素 -> 源体素：v_src = inv(src_affine) @ inv(warp) @ target_affine @ v_dst
    C = np.linalg.inv(src_affine) @ np.linalg.inv(warp) @ target_affine
    out = affine_transform(
        np.asarray(data, dtype=np.float32),
        matrix=C[:3, :3],
        offset=C[:3, 3],
        output_shape=tuple(int(s) for s in target_shape[:3]),
        order=order,
        mode="constant",
        cval=cval,
    )
    return out


def resample_img(img, warp, target_affine, target_shape, order=1, cval=0.0):
    """nibabel 图像版重采样（在 img 上施加 warp）。"""
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    return resample_with_affine(data, img.affine, warp, target_affine, target_shape, order, cval)


# =====================================================================
# 受试者发现 / 元数据解析
# =====================================================================
def sort_vol_key(p):
    """按 vol_XXX 的数字后缀排序。"""
    m = re.search(r"(\d+)$", Path(p).name)
    return int(m.group(1)) if m else 0


def list_subjects(data_root, subjects=None):
    """发现所有 sub-XXXX 目录；subjects 可为 None(全部) 或 id 列表。"""
    data_root = Path(data_root)
    dirs = [d for d in (data_root / "data").glob("sub-*") if d.is_dir()]
    dirs = sorted(dirs, key=lambda d: int(re.sub(r"\D", "", d.name) or 0))
    if subjects:
        wanted = {("sub-" + s) if not str(s).startswith("sub-") else str(s) for s in subjects}
        dirs = [d for d in dirs if d.name in wanted]
    return dirs


def read_participants(meta_dir):
    """读取 participants.tsv，返回 {participant_id: {字段: 值}}。"""
    rows = {}
    p = Path(meta_dir) / "participants.tsv"
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["participant_id"]] = r
    return rows


def parse_protocols_tsv(path):
    """解析转置格式的 MRI_protocols_*.tsv，返回 {protocol_no: {字段: 值}}。"""
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        cells = ln.split("\t")
        if cells and cells[0].strip().startswith("Protocol"):
            header_idx = i
            break
    if header_idx is None:
        return {}
    headers = lines[header_idx].split("\t")
    out = {}
    for ln in lines[header_idx + 1:]:
        cells = ln.split("\t")
        field = cells[0].strip() if cells else ""
        for col_i, h in enumerate(headers):
            h = h.strip()
            if h.isdigit() and col_i < len(cells):
                out.setdefault(int(h), {})[field] = cells[col_i].strip().replace(",", ".")
    return out


def get_tr_map(meta_dir, default_tr=2.5):
    """从 MRI_protocols_rsMRI.tsv 得到 {protocol_no: TR(秒)}，解析失败用默认值。"""
    tr_map = {}
    p = Path(meta_dir) / "MRI_protocols_rsMRI.tsv"
    if p.exists():
        try:
            protos = parse_protocols_tsv(p)
            for no, fields in protos.items():
                for k, v in fields.items():
                    if "TR" in k and "s" in k.lower():
                        m = re.search(r"\d+(?:\.\d+)?", v)
                        if m:
                            tr_map[no] = float(m.group())
        except Exception as e:  # noqa: BLE001
            logging.warning("解析 TR 协议表失败，使用默认 TR=%.2f：%s", default_tr, e)
    return tr_map


def get_slice_order_map(meta_dir):
    """从 MRI 协议表解析每个协议的切片采集顺序。"""
    order_map = {}
    p = Path(meta_dir) / "MRI_protocols_rsMRI.tsv"
    if not p.exists():
        return order_map
    try:
        protos = parse_protocols_tsv(p)
        for no, fields in protos.items():
            for key, value in fields.items():
                normalized_key = key.lower().replace(" ", "")
                if "sliceacquisitionorder" not in normalized_key:
                    continue
                normalized_value = value.lower()
                if "interleave" in normalized_value:
                    order_map[no] = "interleaved"
                elif "descend" in normalized_value:
                    order_map[no] = "descending"
                elif "ascend" in normalized_value:
                    order_map[no] = "ascending"
    except Exception as e:  # noqa: BLE001
        logging.warning("解析切片采集顺序失败，将使用 config.slice_order：%s", e)
    return order_map


def load_any_nifti(path):
    """加载 NIfTI，兼容无扩展名的单文件 NIfTI（如 SRPBS 的 rsfmri/vol_*）。"""
    from nibabel.fileholders import FileHolder
    try:
        return nib.load(str(path))
    except nib.filebasedimages.ImageFileError:
        # 无扩展名的单文件 NIfTI：手动构造文件映射，从同一文件读 header 与 data
        return nib.Nifti1Image.from_file_map({"image": FileHolder(str(path))})


def merge_volumes_4d(vol_paths):
    """把逐时间点 3D NIfTI（vol_XXX）合并为一个 4D NIfTI image。"""
    vol_paths = sorted(vol_paths, key=sort_vol_key)
    imgs = [load_any_nifti(p) for p in vol_paths]
    return nib.concat_images(imgs, check_affines=False)


# =====================================================================
# 去颅骨 / 脑掩膜
# =====================================================================
def skull_strip(data):
    """用 dipy 的 median_otsu 去颅骨；失败则退回强度阈值法。返回 (masked_data, mask)。"""
    try:
        from dipy.segment.mask import median_otsu
        from scipy.ndimage import zoom
        # 低分辨率上做去颅骨（体素减至 1/8，显著加速），再把掩膜插值回原分辨率
        ds = 2
        small = data[::ds, ::ds, ::ds]
        _, mask_small = median_otsu(small, vol_idx=None, median_radius=2, numpass=2)
        mask = zoom(mask_small.astype(np.float32), float(ds), order=1) > 0.5
        mask = mask[tuple(slice(0, s) for s in data.shape)]   # 裁剪回原尺寸（处理奇数维度）
        masked = np.where(mask, data, 0.0)
        return masked, mask
    except Exception as e:  # noqa: BLE001
        logging.warning("median_otsu 失败，退回阈值法：%s", e)
        pos = data[data > 0]
        thr = np.percentile(pos, 10) if pos.size else data.max() * 0.5
        mask = data > thr
        return np.where(mask, data, 0.0), mask


# =====================================================================
# MNI 模板 / 组织概率图 / AAL 图谱
# =====================================================================
def _resample_to_grid(img, target_affine, target_shape, order):
    from nilearn.image import resample_img
    interp = "nearest" if order == 0 else "continuous"
    return resample_img(
        img,
        target_affine=target_affine,
        target_shape=tuple(int(s) for s in target_shape) if target_shape is not None else None,
        interpolation=interp,
    )


def get_mni_templates(cfg):
    """加载（或在线下载）MNI 模板 + GM/WM/CSF 概率图 + 脑掩膜，统一重采样到 cfg.mni_resample_mm 网格。

    返回 dict：{"t1": img, "gm": img, "wm": img, "csf": img, "mask": img}
    """
    from nilearn import datasets

    data_dir = (str(cfg.nilearn_data_dir) if cfg.nilearn_data_dir
                else str(Path(cfg.out_root) / "nilearn_data"))

    def _load(p, key):
        if p and Path(p).exists():
            return nib.load(str(p))
        b = datasets.fetch_icbm152_2009(data_dir=data_dir)
        return nib.load(b[key])

    t1 = _load(cfg.mni_t1, "t1")
    # 以 T1 为基准构造目标网格：体素尺寸 = mni_resample_mm，原点与 T1 对齐
    mm = float(cfg.mni_resample_mm)
    tgt_affine = np.eye(4)
    for i in range(3):
        tgt_affine[i, i] = mm * np.sign(t1.affine[i, i])
    tgt_affine[:3, 3] = t1.affine[:3, 3]
    t1r = _resample_to_grid(t1, tgt_affine, None, 1)

    A = t1r.affine
    S = t1r.shape
    tpl = {"t1": t1r}
    for key, attr, order in [("gm", cfg.mni_gm, 1), ("wm", cfg.mni_wm, 1),
                             ("csf", cfg.mni_csf, 1), ("mask", cfg.mni_mask, 0)]:
        img = _load(attr, key)
        tpl[key] = _resample_to_grid(img, A, S, order)
    return tpl


def _fetch_aal_manual(data_dir):
    """nilearn 下载失败时的兜底：跳过证书校验手动下载 SPM12 AAL（AAL.nii + AAL.xml）。

    返回与 nilearn fetch_atlas_aal 相同字段的 dict：{"maps": 路径, "labels": [...], "indices": [...]}。
    用于处理某些环境下 gin.cnrs.fr 的 SSL 证书校验失败问题。
    """
    import io
    import tarfile
    import xml.etree.ElementTree as ET
    import requests
    import urllib3

    urllib3.disable_warnings()
    url = "https://www.gin.cnrs.fr/AAL_files/aal_for_SPM12.tar.gz"
    out = Path(data_dir) / "aal_SPM12" / "aal"
    out.mkdir(parents=True, exist_ok=True)
    aal_nii = out / "AAL.nii"
    aal_xml = out / "AAL.xml"

    if not (aal_nii.exists() and aal_xml.exists()):
        logging.info("手动下载 AAL（跳过证书校验）：%s", url)
        r = requests.get(url, verify=False, timeout=120)
        r.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
            for m in tar.getmembers():
                base = Path(m.name).name
                if base in ("AAL.nii", "AAL.xml"):
                    m.name = base          # 展平到 out 目录，避免路径穿越
                    tar.extract(m, out)

    labels = ["Background"]
    indices = ["0"]
    tree = ET.parse(str(aal_xml))
    for label in tree.getroot().iter("label"):
        indices.append(label.find("index").text)
        labels.append(label.find("name").text)

    return {"maps": str(aal_nii), "labels": labels, "indices": indices}

def get_aal(cfg, mni_t1=None):
    """加载（或在线下载）AAL 图谱，返回 (aal_img, regions)。

    aal_img : 标签图（整数脑区 ID，0 为背景）
    regions : [(脑区ID, 名称), ...]，供脑区时序提取使用。
    若提供了 mni_t1，则把 AAL 重采样到同一网格（最近邻），保证与归一化 BOLD 对齐。
    """
    from nilearn import datasets

    labels = None
    indices = []
    if cfg.aal_atlas and Path(cfg.aal_atlas).exists():
        aal_img = nib.load(str(cfg.aal_atlas))
        if cfg.aal_labels and Path(cfg.aal_labels).exists():
            labels = [ln.strip() for ln in open(cfg.aal_labels, encoding="utf-8") if ln.strip()]
    else:
        data_dir = (str(cfg.nilearn_data_dir) if cfg.nilearn_data_dir
                    else str(Path(cfg.out_root) / "nilearn_data"))
        try:
            b = datasets.fetch_atlas_aal(version="SPM12", data_dir=data_dir)
        except Exception as e:  # noqa: BLE001
            logging.warning("nilearn 自动下载 AAL 失败（%s），改用跳过证书校验的方式下载...", e)
            b = _fetch_aal_manual(data_dir)
        maps = b["maps"] if "maps" in b else b.get("maps")
        labels = list(b["labels"]) if "labels" in b else b.get("labels", [])
        indices = list(b["indices"]) if "indices" in b else b.get("indices", [])
        # 新版 nilearn(>=0.13)：maps 是单个 NIfTI 路径；旧版：是每个脑区一张图的列表
        if isinstance(maps, (str, Path)):
            aal_img = nib.load(str(maps))
        else:
            ref = maps[0]
            lab = np.zeros(ref.shape, dtype=np.int16)
            for i, m in enumerate(maps):
                d = np.asarray(m.get_fdata(dtype=np.float32), dtype=np.float32)
                lab[d > 0] = i + 1
            aal_img = nib.Nifti1Image(lab, ref.affine)
            indices = []

    # 组装 regions：用 indices 把标签名映射到图像中的整数脑区 ID
    if labels and indices and len(labels) == len(indices):
        regions = []
        for name, idx in zip(labels, indices):
            v = int(idx)
            if v == 0:
                continue
            regions.append((v, name))
    elif labels:
        # 本地图谱常使用非连续 ID（标准 AAL 例如 2001、2002...），不能假定为 1..N。
        uniq = sorted({int(x) for x in np.asarray(aal_img.dataobj).ravel() if int(x) != 0})
        local_labels = list(labels)
        if len(local_labels) == len(uniq) + 1 and local_labels[0].lower() in {
                "background", "背景"}:
            local_labels = local_labels[1:]
        if len(local_labels) != len(uniq):
            raise ValueError(
                f"AAL 标签数({len(local_labels)})与图谱非零 ID 数({len(uniq)})不一致"
            )
        regions = list(zip(uniq, local_labels))
    else:
        uniq = sorted({int(x) for x in np.asarray(aal_img.dataobj).ravel() if int(x) != 0})
        regions = [(v, "region_" + str(v)) for v in uniq]

    if mni_t1 is not None:
        aal_img = _resample_to_grid(aal_img, mni_t1.affine, mni_t1.shape, 0)
    return aal_img, regions


# =====================================================================
# 每受试者输出路径
# =====================================================================
@dataclass
class SubjPaths:
    """一个受试者的所有输入/中间/输出文件路径。"""
    subj_id: str
    out_dir: Path
    src_dir: Path = None      # 原始数据目录 data/sub-XXXX（仅 merge 步骤用）

    # 输入（原始 NIfTI，来自数据集）
    bold_4d: Path = None      # 合并后的 4D BOLD
    t1: Path = None           # 结构像
    t1_brain: Path = None     # 去颅骨结构像
    brain_mask: Path = None   # 脑掩膜

    # 预处理中间产物
    bold_st: Path = None      # 切片时间校正后
    bold_mc: Path = None      # 头动校正后
    mc_params: Path = None    # 6 列头动参数 (T x 6)
    mc_fd: Path = None        # 帧间位移 FD (T,)
    bold_norm: Path = None    # 空间归一化后（MNI）
    bold_smooth: Path = None  # 空间平滑后
    bold_clean: Path = None   # 去噪后

    # 配准变换 / 掩膜
    bold2t1: object = None    # BOLD→T1 ANTs 变换文件列表
    t12mni: object = None     # T1→MNI ANTs 变换文件列表
    gm_mask: Path = None      # GM 掩膜（BOLD 空间，脑区分割产物）
    wm_mask: Path = None      # WM 掩膜（BOLD 空间）
    csf_mask: Path = None     # CSF 掩膜（BOLD 空间）

    # 脑区时序矩阵
    aal: Path = None          # BOLD 空间 AAL 标签图
    ts: Path = None           # (116, T) 脑区时序矩阵

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.mc_params = self.out_dir / "func" / f"{self.subj_id}_mc_params.tsv"
        self.mc_fd = self.out_dir / "func" / f"{self.subj_id}_mc_fd.tsv"

    def restore_existing(self):
        """从输出目录恢复已有产物路径，供分步执行和断点续跑使用。"""
        sid = self.subj_id
        candidates = {
            "bold_4d": self.out_dir / "func" / f"{sid}_bold.nii.gz",
            "t1": self.out_dir / "anat" / f"{sid}_T1w.nii.gz",
            "t1_brain": self.out_dir / "anat" / f"{sid}_T1w_brain.nii.gz",
            "brain_mask": self.out_dir / "masks" / f"{sid}_brain_mask.nii.gz",
            "bold_st": self.out_dir / "func" / f"{sid}_st.nii.gz",
            "bold_mc": self.out_dir / "func" / f"{sid}_mc.nii.gz",
            "bold_norm": self.out_dir / "func" / f"{sid}_norm.nii.gz",
            "bold_smooth": self.out_dir / "func" / f"{sid}_smooth.nii.gz",
            "bold_clean": self.out_dir / "func" / f"{sid}_clean.nii.gz",
            "gm_mask": self.out_dir / "masks" / f"{sid}_gm_mask.nii.gz",
            "wm_mask": self.out_dir / "masks" / f"{sid}_wm_mask.nii.gz",
            "csf_mask": self.out_dir / "masks" / f"{sid}_csf_mask.nii.gz",
            "aal": self.out_dir / "parcellation" / f"{sid}_aal116.nii.gz",
            "ts": self.out_dir / "parcellation" / f"{sid}_aal116_ts.npy",
        }
        for field_name, candidate in candidates.items():
            if candidate.exists():
                setattr(self, field_name, candidate)
        reg_dir = self.out_dir / "reg"
        bold2t1_files = sorted(reg_dir.glob(f"{sid}_bold2t1_*"))
        t12mni_files = sorted(reg_dir.glob(f"{sid}_t12mni_*"))
        if bold2t1_files:
            self.bold2t1 = tuple(bold2t1_files)
        if t12mni_files:
            self.t12mni = tuple(t12mni_files)
        return self

    def require(self, *field_names):
        """检查步骤所需的已有产物，并返回对应路径列表。"""
        missing = []
        values = []
        for field_name in field_names:
            value = getattr(self, field_name)
            if value is None or not Path(value).exists():
                missing.append(field_name)
            values.append(value)
        if missing:
            raise FileNotFoundError(
                f"[{self.subj_id}] 缺少前置产物：{', '.join(missing)}；"
                "请先执行所需的前序步骤"
            )
        return values

    def remove_intermediate_images(self):
        """删除可重新生成的中间 4D 图像，保留最终结果、掩膜和 QC 参数。"""
        removed = []
        for field_name in ("bold_st", "bold_mc", "bold_norm"):
            value = getattr(self, field_name)
            if value is not None and Path(value).exists():
                Path(value).unlink()
                setattr(self, field_name, None)
                removed.append(str(value))
        return removed
