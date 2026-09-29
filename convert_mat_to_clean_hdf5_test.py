# -*- coding: utf-8 -*-
"""
Convert the inspected MATLAB preprocessed EEG file into the clean HDF5 format
expected by ws.py.

This script is intentionally based on reports/mat_hdf5_structure_report.md:

    MAT file:
        data/dataset1_processeddata_Sub-001_sess-01_task-MI_eeg.mat (example)
    MAT data field:
        data, shape (58, 1000, 200), float32
    MAT label field:
        labels, shape (1, 200), uint8
    Target HDF5 format:
        clean_eeg: (epochs, channels, times)
        labels: (epochs,)

The report indicates that labels length 200 matches data axis 2, so conversion
uses np.transpose(data, (2, 0, 1)).

Run a no-write validation first:

    python convert_mat_to_clean_hdf5.py --dry-run

Write the converted file:

    python convert_mat_to_clean_hdf5.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

import h5py
import numpy as np
import scipy.io as sio
import config


# =========================
# Report-derived configuration
# =========================

MAT_INPUT_PATH = Path(config.MAT_INPUT_PATH)
OUTPUT_HDF5_PATH = Path(config.MAT_OUTPUT_PATH)

DATA_FIELD_PATH = "data"
LABEL_FIELD_PATH = "labels"

# From the report:
#   data:   (58, 1000, 200)
#   labels: (1, 200)
# Therefore axis 2 is epochs. The classifier expects (epochs, channels, times).
EXPECTED_SOURCE_DATA_SHAPE = (58, 1000, 200)
EXPECTED_SOURCE_LABEL_SHAPE = (1, 200)
TRANSPOSE_AXES = (2, 0, 1)
SOURCE_AXIS_ORDER = "channels,times,epochs"
TARGET_AXIS_ORDER = "epochs,channels,times"

# The MAT report did not include sampling rate or channel names. These defaults
# follow the clean HDF5 reference file in the report.
SAMPLING_RATE = 250.0
UNIT = "microvolts"
CHANNEL_NAME_PREFIX = "Ch"

# Keep the output close to the current artifact_remove.py clean HDF5 format.
CLEAN_EEG_DTYPE = np.float32
LABEL_DTYPE = np.int32
OVERWRITE = False


def main() -> None:
    args = parse_args()

    # 批量输入范围由本地配置指定，不在脚本中保存个人目录或实验范围。
    base_dir = Path(config.MAT_DATASET_DIR)
    start_sub = config.MAT_BATCH_START_SUB
    end_sub = config.MAT_BATCH_END_SUB
    n_sessions = config.MAT_BATCH_N_SESSIONS
    if start_sub < 1 or end_sub < start_sub or n_sessions < 1:
        raise ValueError("Invalid MAT batch subject/session range")
    total = (end_sub - start_sub + 1) * n_sessions

    completed = 0
    success_count = 0
    skip_count = 0

    print("=" * 70)
    print(f"开始批量转换：Sub-{start_sub:03d} 到 Sub-{end_sub:03d}，共 {total} 个文件")
    print("=" * 70)

    for sub_idx in range(start_sub, end_sub + 1):
        sub_id = f"{sub_idx:03d}"
        for sess_idx in range(1, n_sessions + 1):
            sess_id = f"{sess_idx:02d}"

            mat_filename = f"dataset1_processeddata_Sub-{sub_id}_sess-{sess_id}_task-MI_eeg.mat"
            hdf5_filename = f"clean_dataset1_processeddata_Sub-{sub_id}_sess-{sess_id}_task-MI_eeg.hdf5"

            mat_path = base_dir / mat_filename
            output_path = base_dir / hdf5_filename

            # 文件不存在则跳过
            if not mat_path.exists():
                completed += 1
                skip_count += 1
                _print_progress(completed, total, sub_id, sess_id, "[MAT不存在]")
                continue

            # 动态修改全局变量，供 write_clean_hdf5 等函数内部引用
            global MAT_INPUT_PATH, OUTPUT_HDF5_PATH
            MAT_INPUT_PATH = mat_path
            OUTPUT_HDF5_PATH = output_path

            try:
                # 核心转换逻辑（原封不动）
                data, labels = load_reported_mat_fields(MAT_INPUT_PATH)
                clean_eeg, clean_labels = convert_to_clean_arrays(data, labels)
                validate_clean_arrays(clean_eeg, clean_labels)

                if args.dry_run:
                    completed += 1
                    success_count += 1
                    _print_progress(completed, total, sub_id, sess_id, "[Dry-run OK]")
                    continue

                write_clean_hdf5(
                    OUTPUT_HDF5_PATH,
                    clean_eeg,
                    clean_labels,
                    overwrite=args.overwrite or OVERWRITE,
                )
                verify_written_hdf5(OUTPUT_HDF5_PATH, clean_eeg, clean_labels)
                success_count += 1

                completed += 1
                _print_progress(completed, total, sub_id, sess_id, "[完成]")

            except Exception as e:
                # 任何转换异常（shape不符、字段缺失、验证失败等）都跳过该文件
                completed += 1
                skip_count += 1
                _print_progress(completed, total, sub_id, sess_id, f"[跳过:{type(e).__name__}]")
                continue

    print(f"\n{'=' * 70}")
    print(f"批量转换完成：成功 {success_count}/{total}，跳过 {skip_count}/{total}")
    print(f"{'=' * 70}")


def _print_progress(completed, total, sub_id, sess_id, status=""):
    """辅助函数：刷新文本进度条"""
    progress = completed / total
    bar_len = 30
    filled = int(bar_len * progress)
    bar = "█" * filled + "-" * (bar_len - filled)
    sys.stdout.write(
        f"\r进度: [{bar}] {completed:>3}/{total} ({progress*100:>5.1f}%) | "
        f"Sub-{sub_id} Sess-{sess_id} {status}"
    )
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the inspected MATLAB EEG data into ws.py clean HDF5 format."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, transpose, and validate arrays without writing an HDF5 file.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=f"Optional output path. Default: {OUTPUT_HDF5_PATH}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output HDF5 file.",
    )
    return parser.parse_args()


def load_reported_mat_fields(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not mat_path.is_file():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mat = sio.loadmat(
        mat_path,
        struct_as_record=False,
        squeeze_me=False,
        chars_as_strings=True,
    )

    if DATA_FIELD_PATH not in mat:
        raise KeyError(f"Missing MAT data field: {DATA_FIELD_PATH}")
    if LABEL_FIELD_PATH not in mat:
        raise KeyError(f"Missing MAT label field: {LABEL_FIELD_PATH}")

    data = np.asarray(mat[DATA_FIELD_PATH])
    labels = np.asarray(mat[LABEL_FIELD_PATH])

    if data.shape != EXPECTED_SOURCE_DATA_SHAPE:
        raise ValueError(
            f"Unexpected data shape for {DATA_FIELD_PATH}: "
            f"{data.shape} != {EXPECTED_SOURCE_DATA_SHAPE}"
        )
    if labels.shape != EXPECTED_SOURCE_LABEL_SHAPE:
        raise ValueError(
            f"Unexpected labels shape for {LABEL_FIELD_PATH}: "
            f"{labels.shape} != {EXPECTED_SOURCE_LABEL_SHAPE}"
        )
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"{DATA_FIELD_PATH} must be numeric, got {data.dtype}")
    if not np.issubdtype(labels.dtype, np.number):
        raise TypeError(f"{LABEL_FIELD_PATH} must be numeric, got {labels.dtype}")

    return data, labels


def convert_to_clean_arrays(
    source_data: np.ndarray,
    source_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    clean_eeg = np.transpose(source_data, TRANSPOSE_AXES).astype(CLEAN_EEG_DTYPE, copy=False)
    labels = source_labels.reshape(-1).astype(LABEL_DTYPE, copy=False)
    return clean_eeg, labels


def validate_clean_arrays(clean_eeg: np.ndarray, labels: np.ndarray) -> None:
    if clean_eeg.ndim != 3:
        raise ValueError(f"clean_eeg must be 3D, got shape {clean_eeg.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if clean_eeg.shape[0] != labels.shape[0]:
        raise ValueError(
            "Epoch count and label count do not match: "
            f"{clean_eeg.shape[0]} != {labels.shape[0]}"
        )
    if not np.all(np.isfinite(clean_eeg)):
        raise ValueError("clean_eeg contains NaN or Inf values.")
    if not np.all(np.isfinite(labels)):
        raise ValueError("labels contains NaN or Inf values.")

    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, np.array([1, 2], dtype=LABEL_DTYPE)):
        raise ValueError(
            "Expected labels to contain exactly classes [1, 2], "
            f"got {unique_labels.tolist()}"
        )


def write_clean_hdf5(
    output_path: Path,
    clean_eeg: np.ndarray,
    labels: np.ndarray,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use --overwrite if you intentionally want to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel_names = np.array(
        [f"{CHANNEL_NAME_PREFIX}{idx + 1}" for idx in range(clean_eeg.shape[1])],
        dtype="S",
    )

    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("clean_eeg", data=clean_eeg.astype(CLEAN_EEG_DTYPE, copy=False))
        h5_file.create_dataset("labels", data=labels.astype(LABEL_DTYPE, copy=False))
        h5_file.create_dataset("channel_names", data=channel_names)
        h5_file.attrs["sampling_rate"] = SAMPLING_RATE
        h5_file.attrs["unit"] = UNIT
        h5_file.attrs["source_mat_path"] = str(MAT_INPUT_PATH)
        h5_file.attrs["source_data_field"] = DATA_FIELD_PATH
        h5_file.attrs["source_label_field"] = LABEL_FIELD_PATH
        h5_file.attrs["source_data_shape"] = str(EXPECTED_SOURCE_DATA_SHAPE)
        h5_file.attrs["source_label_shape"] = str(EXPECTED_SOURCE_LABEL_SHAPE)
        h5_file.attrs["source_axis_order"] = SOURCE_AXIS_ORDER
        h5_file.attrs["target_axis_order"] = TARGET_AXIS_ORDER
        h5_file.attrs["transpose_axes"] = str(TRANSPOSE_AXES)
        h5_file.attrs["channel_names_source"] = "generated sequential names"
        h5_file.attrs["generated_by"] = Path(__file__).name
        h5_file.attrs["generated_at"] = datetime.now().isoformat(timespec="seconds")


def verify_written_hdf5(
    output_path: Path,
    expected_eeg: np.ndarray,
    expected_labels: np.ndarray,
) -> None:
    with h5py.File(output_path, "r") as h5_file:
        required_keys = {"clean_eeg", "labels", "channel_names"}
        missing = required_keys.difference(h5_file.keys())
        if missing:
            raise KeyError(f"Converted HDF5 is missing datasets: {sorted(missing)}")

        clean_eeg = h5_file["clean_eeg"][:]
        labels = h5_file["labels"][:]

        if clean_eeg.shape != expected_eeg.shape:
            raise ValueError(f"Written clean_eeg shape mismatch: {clean_eeg.shape}")
        if labels.shape != expected_labels.shape:
            raise ValueError(f"Written labels shape mismatch: {labels.shape}")
        if clean_eeg.dtype != np.dtype(CLEAN_EEG_DTYPE):
            raise TypeError(f"Written clean_eeg dtype mismatch: {clean_eeg.dtype}")
        if labels.dtype != np.dtype(LABEL_DTYPE):
            raise TypeError(f"Written labels dtype mismatch: {labels.dtype}")
        if not np.array_equal(clean_eeg, expected_eeg):
            raise ValueError("Written clean_eeg does not exactly match converted data.")
        if not np.array_equal(labels, expected_labels):
            raise ValueError("Written labels do not exactly match converted labels.")
        if h5_file.attrs.get("sampling_rate") != SAMPLING_RATE:
            raise ValueError("Written sampling_rate attribute mismatch.")
        if h5_file.attrs.get("unit") != UNIT:
            raise ValueError("Written unit attribute mismatch.")


def print_conversion_summary(
    source_data: np.ndarray,
    source_labels: np.ndarray,
    clean_eeg: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    label_distribution = {int(label): int(count) for label, count in zip(unique, counts)}

    print("MAT -> clean HDF5 conversion summary")
    print(f"  MAT input: {MAT_INPUT_PATH}")
    print(f"  Output HDF5: {output_path}")
    print(f"  Source data: {DATA_FIELD_PATH}, shape={source_data.shape}, dtype={source_data.dtype}")
    print(f"  Source labels: {LABEL_FIELD_PATH}, shape={source_labels.shape}, dtype={source_labels.dtype}")
    print(f"  Transpose axes: {TRANSPOSE_AXES}")
    print(f"  clean_eeg: shape={clean_eeg.shape}, dtype={clean_eeg.dtype}")
    print(f"  labels: shape={labels.shape}, dtype={labels.dtype}, distribution={label_distribution}")
    print(f"  clean_eeg stats: {format_numeric_stats(clean_eeg)}")


def format_numeric_stats(array: np.ndarray) -> str:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return "no finite values"
    return (
        f"min={float(np.min(finite)):.6g}, "
        f"max={float(np.max(finite)):.6g}, "
        f"mean={float(np.mean(finite)):.6g}, "
        f"std={float(np.std(finite)):.6g}"
    )


if __name__ == "__main__":
    main()