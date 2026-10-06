# SRPBS 预处理流水线（Python + ANTs + 可选 GPU）

本目录实现了 SRPBS Open 数据集的预处理流程，工具链为 **nibabel + nilearn + dipy + antspyx**，
无需 FSL/SPM。去噪回归、滤波和脑区均值可选走 **cupy(GPU)**，未安装时自动回退 CPU。

## 1. 依赖安装

```bash
pip install -r requirements.txt

# 可选：GPU 加速（根据本机 CUDA 版本选择，如 CUDA 12）
pip install cupy-cuda12x
```

## 2. 流程概述

| 步骤 | 文件 | 说明 |
|---|---|---|
| (可选) DICOM→NIfTI | `step0_dcm2niix.py` | 仅当持有原始 DICOM 时才用；本数据集已是 NIfTI，通常跳过 |
| ① 合并 4D | `step1_merge_4d.py` | 把 `rsfmri/vol_*` 逐时间点 3D EPI 合并为 4D BOLD；T1 去颅骨 |
| ② 预处理 | `step2_preprocess.py` | 切片时间校正 → 头动校正 → BOLD→T1 配准 → T1→MNI 归一化 → 平滑 |
| ③ 去噪 | `step3_denoise.py` | 去趋势 + 带通滤波 + 噪声回归（头动/WM/CSF） |
| ④ 脑区时序 | `step4_parcellate.py` | AAL116 配准 + 体素均值 → (116, T) 脑区时序矩阵 |

对应你描述的流程：
原始 DICOM --dcm2niix--> 4D NIfTI（= 本数据集已提供的 vol_*，见步骤①）
→ 时间步长校正 / 头动校正 / 结构配准 / 脑区分割 / 空间归一化 / 空间平滑
→ 去趋势滤波 / 噪声回归 → 4D BOLD
→ 配准 AAL116 → 体素均值 → 脑区时序矩阵。

## 3. 运行

```bash
# 全流程、全部受试者（1410 个）
python run_pipeline.py

# 先用 2 个受试者试跑，验证环境
python run_pipeline.py --subjects 0001 0002

# 只跑到某一步
python run_pipeline.py --subjects 0001 --steps merge preprocess

# 启用 GPU
python run_pipeline.py --gpu

# 调整常用参数
python run_pipeline.py --fwhm 8 --low-pass 0.08 --high-pass 0.008

# 4 个受试者并行（并行时使用 CPU）
python run_pipeline.py --n-jobs 4 --gpu off

# SyN 归一化并为 FD>0.5 mm 的时间点加入 spike 回归量
python run_pipeline.py --normalize-dof syn --scrub --fd-thr 0.5
```

所有可调参数集中在 `config.py`（也可命令行覆盖部分参数）。分步执行会自动从输出目录恢复前序产物；
缺少前置文件时会明确报错，不会静默省略处理项。预处理会记录输入和参数签名；代码升级、输入或关键参数
变化后会自动重算，只有签名一致时才复用中间结果。

## 4. 输出目录结构（derivatives/sub-XXXX/）

```
sub-XXXX/
├── func/   {id}_bold.nii.gz, _st.nii.gz, _mc.nii.gz, _norm.nii.gz, _smooth.nii.gz, _clean.nii.gz
│           {id}_mc_params.tsv（6 列头动参数）, {id}_mc_fd.tsv（帧间位移）
├── anat/   {id}_T1w.nii.gz, _T1w_brain.nii.gz, _T1w_norm.nii.gz
├── reg/    {id}_bold2t1_0GenericAffine.mat
│           {id}_t12mni_0GenericAffine.mat, _1Warp.nii.gz, _1InverseWarp.nii.gz（取决于归一化模型）
├── masks/  {id}_brain_mask.nii.gz, _wm_mask.nii.gz, _csf_mask.nii.gz
└── parcellation/
            {id}_aal116.nii.gz          # BOLD 空间 AAL 标签图
            {id}_aal116_ts.npy          # (116, T) 脑区时序矩阵
            {id}_aal116_ts.csv          # 带脑区名的矩阵
            {id}_aal116_labels.csv      # 116 个脑区名称
```

## 5. 实现说明与注意事项

1. **TR 自动识别**：按 `participants.tsv` 的 protocol 字段，从 `MRI_protocols_rsMRI.tsv` 解析每人的 TR
   （协议 2/3/7 为 2.0s、协议 4 为 2.7s、其余 2.5s），无法解析时用 `config.tr_default`。
2. **切片时间校正**：按协议表自动识别升序/交错采集，以 TR/2 为参考点做傅里叶相位平移；
   协议表无法解析时使用 `config.slice_order`。
3. **头动校正**：逐卷刚体(6 参数)配准到参考卷（scipy Powell 优化，多分辨率），并输出 FD 供质控。
4. **结构配准/归一化**：BOLD→T1 使用 ANTs 刚体配准；T1→MNI 可选 `rigid`、`affine` 或
   `syn`（默认 SyNRA），变换文件持久化在 `reg/` 目录。
5. **脑区分割**：不依赖 FSL FAST。用 MNI 组织概率图（GM/WM/CSF）经逆变换映射到 BOLD 空间，
   阈值 0.5 得到 WM/CSF 掩膜用于噪声回归。
6. **去噪**：先对数据和混杂变量执行相同的去趋势与 0.01–0.1 Hz 带通滤波，再回归
   [头动参数 + 头动一阶差分 + WM/CSF 均值]，最后每体素 z 标准化（可关）。WM/CSF 信号从未平滑的
   `norm` 数据提取；`--scrub` 使用 spike 回归量并保持时间轴长度。
7. **GPU**：`--gpu` 时去噪的 lstsq/FFT 与脑区 one-hot 均值在 cupy 上计算；`--gpu off` 强制 CPU；
   默认 `auto`。
8. **首次运行需联网**：MNI 模板（ICBM152 2009）与 AAL 图谱由 nilearn 在线下载并缓存；
   也可在 `config.py` 里指定本地文件路径离线使用。
9. **内存/显存**：归一化网格默认重采样到 3mm（`config.mni_resample_mm`），显著降低显存占用；
   小显存可调大到 4mm 或关闭 GPU。

## 6. 已知局限

- 头动校正为 scipy Powell 优化，速度慢于 FSL MCFLIRT（尤其全量 1410 人时）。
- 组织分割为“模板概率图反投影”的近似，非受试者个体分割。
- 频域滤波使用矩形频率窗，时间序列边界可能出现振铃。
- 当前流程没有使用数据集中的场图进行 EPI 磁敏感畸变校正。

建议：先用 `--subjects 0001 0002` 小样本跑通并检查输出图像质量，再全量运行。

## 7. 常见问题

- **下载 MNI 模板 / AAL 时报 SSL 证书错误**（如 `CERTIFICATE_VERIFY_FAILED`，常见于公司代理/内网环境）：
  1) 手动下载：
     - MNI ICBM152 2009：https://osf.io/7pj92/download （解压出 mni_icbm152_*.nii.gz 系列）
     - AAL SPM12：https://www.gin.cnrs.fr/AAL_files/aal_for_SPM12.tar.gz （解压出 AAL.nii）
  2) 在 `config.py` 中设置本地路径，例如：
     ```python
     mni_t1  = Path(r"...\mni_icbm152_t1_tal_nlin_sym_09a.nii.gz")
     mni_gm  = Path(r"...\mni_icbm152_gm_tal_nlin_sym_09a.nii.gz")
     mni_wm  = Path(r"...\mni_icbm152_wm_tal_nlin_sym_09a.nii.gz")
     mni_csf = Path(r"...\mni_icbm152_csf_tal_nlin_sym_09a.nii.gz")
     mni_mask= Path(r"...\mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz")
     aal_atlas = Path(r"...\AAL.nii")
     ```
     也可临时安装证书：`pip install pip-system-certs` 后重跑。

- **GPU 未生效**：确认已 `pip install cupy-cuda12x`（版本与本机 CUDA 匹配），
  运行 `python -c "import cupy; print(cupy.__version__)"` 验证；`auto` 模式会警告并回退 CPU，
  `--gpu on` 则会直接报错。

- **内存/显存不足**：调大 `config.mni_resample_mm`（如 4.0）或 `--gpu off`。

- **想快速试跑**：`python run_pipeline.py --subjects 0001 0002`（仅两个受试者）。
