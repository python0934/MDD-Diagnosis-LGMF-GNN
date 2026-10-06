"""step0_dcm2niix.py - 可选的 DICOM -> NIfTI 转换。

注意：SRPBS Open 数据集已经是 NIfTI 格式，通常不需要这一步。
仅当你持有原始 DICOM 时才使用本脚本：它调用 dcm2niix 命令行工具，
把每个受试者的 DICOM 目录转换为 NIfTI（4D BOLD 与 3D T1w）。

用法：
    python step0_dcm2niix.py --input DICOM目录 --output 输出目录
"""
from __future__ import annotations

import sys
import shutil
import subprocess
import argparse
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def find_dcm2niix():
    exe = shutil.which("dcm2niix")
    if exe is None:
        raise RuntimeError(
            "未找到 dcm2niix，请先安装并加入 PATH（https://github.com/rordenlab/dcm2niix）"
        )
    return exe


def convert(subj_input_dir, out_dir, exe=None):
    """把一个受试者目录内的 DICOM 转为 NIfTI（gzip 压缩）。"""
    exe = exe or find_dcm2niix()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-z", "y", "-f", "%p_%s", "-o", str(out_dir), str(subj_input_dir)]
    subprocess.run(cmd, check=True)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DICOM -> NIfTI（dcm2niix）")
    ap.add_argument("--input", required=True, help="DICOM 目录")
    ap.add_argument("--output", required=True, help="输出目录")
    args = ap.parse_args()
    convert(args.input, args.output)
    print("完成 ->", args.output)