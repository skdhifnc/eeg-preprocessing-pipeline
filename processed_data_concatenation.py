# -*- coding: utf-8 -*-
"""
已处理 EEG HDF5 数据拼接脚本。

这个脚本只处理已经转换/清洗/分段后的 HDF5 文件，不处理原始连续信号。
输入文件必须是 clean 数据格式：
    clean_eeg:      (epochs, channels, times)
                    第 0 维是试次数/片段数，第 1 维是通道数，第 2 维是每段采样点数
    labels:         (epochs,)
                    每个 epoch 对应一个标签
    channel_names:  (channels,)
                    通道名称列表

脚本默认从 config.PROCESSED_CONCAT_INPUT_FILES 读取输入路径，
并写入 config.PROCESSED_CONCAT_OUTPUT_PATH。
拼接方式是沿 clean_eeg 的第 0 维，也就是 epoch 维度纵向拼接。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np

import config


# 每个 processed HDF5 文件必须包含的三个核心数据集。
REQUIRED_DATASETS = ("clean_eeg", "labels", "channel_names")


class ProcessedFileInfo(NamedTuple):
    """保存单个 processed HDF5 文件的结构信息，方便后续统一校验。"""

    path: Path
    epochs: int
    channels: int
    times: int
    eeg_dtype: np.dtype
    label_dtype: np.dtype
    channel_names: np.ndarray
    sampling_rate: object
    unit: object


def parse_args() -> argparse.Namespace:
    """解析命令行参数：支持只检查不写入，以及允许覆盖已有输出文件。"""

    parser = argparse.ArgumentParser(
        description="Concatenate processed HDF5 files containing clean_eeg and labels."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the expected output without writing a file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing the output file if it already exists.",
    )
    return parser.parse_args()


def attrs_equal(left: object, right: object) -> bool:
    """比较 HDF5 属性值；采样率这类浮点数使用 np.isclose 避免精度误差。"""

    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return bool(np.isclose(left, right))
        except TypeError:
            return False
    return left == right


def inspect_processed_file(path: Path) -> ProcessedFileInfo:
    """
    检查单个 processed HDF5 文件是否符合 clean 数据格式。

    这里只读取结构信息和少量元数据，不把完整 clean_eeg 加载进内存。
    """

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with h5py.File(path, "r") as h5_file:
        # 先检查核心数据集是否齐全，缺任何一个都不能进行 processed 数据拼接。
        missing = [name for name in REQUIRED_DATASETS if name not in h5_file]
        if missing:
            raise KeyError(f"{path}: missing datasets: {missing}")

        clean_eeg = h5_file["clean_eeg"]
        labels = h5_file["labels"]
        channel_names = h5_file["channel_names"]

        # clean_eeg 的约定形状是 (epochs, channels, times)。
        if clean_eeg.ndim != 3:
            raise ValueError(f"{path}: clean_eeg must be 3D, got shape {clean_eeg.shape}")
        # labels 必须是一维数组，且长度要等于 epoch 数。
        if labels.ndim != 1:
            raise ValueError(f"{path}: labels must be 1D, got shape {labels.shape}")
        # channel_names 必须是一维数组，且长度要等于通道数。
        if channel_names.ndim != 1:
            raise ValueError(
                f"{path}: channel_names must be 1D, got shape {channel_names.shape}"
            )

        epochs, channels, times = clean_eeg.shape
        if labels.shape[0] != epochs:
            raise ValueError(
                f"{path}: labels length must match epoch count, "
                f"got {labels.shape[0]} != {epochs}"
            )
        if channel_names.shape[0] != channels:
            raise ValueError(
                f"{path}: channel_names length must match channel count, "
                f"got {channel_names.shape[0]} != {channels}"
            )

        # 返回文件摘要信息，后面会用它和其他输入文件做一致性检查。
        return ProcessedFileInfo(
            path=path,
            epochs=int(epochs),
            channels=int(channels),
            times=int(times),
            eeg_dtype=np.dtype(clean_eeg.dtype),
            label_dtype=np.dtype(labels.dtype),
            channel_names=channel_names[:],
            sampling_rate=h5_file.attrs.get("sampling_rate"),
            unit=h5_file.attrs.get("unit"),
        )


def validate_compatible_inputs(paths: list[Path]) -> list[ProcessedFileInfo]:
    """
    检查所有输入文件是否可以拼接。

    processed 数据只能沿 epoch 维拼接，所以通道数、时间点数、通道名、
    数据类型和采样率必须一致。
    """

    if len(paths) < 2:
        raise ValueError("Provide at least 2 processed HDF5 input files.")

    infos = [inspect_processed_file(path) for path in paths]
    first = infos[0]

    # 采样率是后续滤波/分类的重要元信息，要求所有输入都存在并保持一致。
    if first.sampling_rate is None:
        raise ValueError(f"{first.path}: missing sampling_rate attribute")

    for info in infos[1:]:
        # 所有文件的 clean_eeg 后两维必须一致，只有第 0 维 epochs 可以不同。
        if info.channels != first.channels:
            raise ValueError(
                f"{info.path}: channel count mismatch {info.channels} != {first.channels}"
            )
        if info.times != first.times:
            raise ValueError(f"{info.path}: time length mismatch {info.times} != {first.times}")
        if info.eeg_dtype != first.eeg_dtype:
            raise ValueError(
                f"{info.path}: clean_eeg dtype mismatch {info.eeg_dtype} != {first.eeg_dtype}"
            )
        if info.label_dtype != first.label_dtype:
            raise ValueError(
                f"{info.path}: labels dtype mismatch {info.label_dtype} != {first.label_dtype}"
            )
        # 通道名称和顺序也必须一致，否则同一列数据可能对应不同电极。
        if not np.array_equal(info.channel_names, first.channel_names):
            raise ValueError(f"{info.path}: channel_names do not match the first input file")
        if info.sampling_rate is None:
            raise ValueError(f"{info.path}: missing sampling_rate attribute")
        if not attrs_equal(info.sampling_rate, first.sampling_rate):
            raise ValueError(
                f"{info.path}: sampling_rate mismatch "
                f"{info.sampling_rate!r} != {first.sampling_rate!r}"
            )

    return infos


def print_summary(infos: list[ProcessedFileInfo], output_path: Path) -> None:
    """打印拼接前后的数据规模，方便 dry-run 时确认结果是否符合预期。"""

    total_epochs = sum(info.epochs for info in infos)
    first = infos[0]

    print("Processed HDF5 concatenation summary")
    for info in infos:
        print(
            f"  Input: {info.path} | clean_eeg=({info.epochs}, "
            f"{info.channels}, {info.times}) | labels=({info.epochs},)"
        )
    print(f"  Output: {output_path}")
    print(f"  Output clean_eeg shape: ({total_epochs}, {first.channels}, {first.times})")
    print(f"  Output labels shape: ({total_epochs},)")
    print(f"  Sampling rate: {first.sampling_rate}")
    print(f"  Unit: {first.unit}")


def concatenate_processed_hdf5(
    input_paths: list[Path],
    output_path: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """
    拼接多个 processed HDF5 文件并写入新的 HDF5 文件。

    核心逻辑：
    1. 先校验每个输入文件结构和元数据是否兼容。
    2. 创建输出数据集，大小为所有输入 epoch 数之和。
    3. 逐个文件读取 clean_eeg 和 labels，写入输出文件的对应区间。
    """

    infos = validate_compatible_inputs(input_paths)
    print_summary(infos, output_path)

    if dry_run:
        print("Dry run complete. No HDF5 file was written.")
        return

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    first = infos[0]
    total_epochs = sum(info.epochs for info in infos)

    with h5py.File(output_path, "w") as out_file:
        # 预先创建完整大小的数据集，避免先把所有输入拼到内存里。
        out_eeg = out_file.create_dataset(
            "clean_eeg",
            shape=(total_epochs, first.channels, first.times),
            dtype=first.eeg_dtype,
            chunks=(min(64, total_epochs), first.channels, first.times),
        )
        out_labels = out_file.create_dataset(
            "labels",
            shape=(total_epochs,),
            dtype=first.label_dtype,
            chunks=(min(1024, total_epochs),),
        )
        # 通道名称、采样率和单位使用第一个输入文件作为模板。
        out_file.create_dataset("channel_names", data=first.channel_names)
        out_file.attrs["sampling_rate"] = first.sampling_rate
        if first.unit is not None:
            out_file.attrs["unit"] = first.unit

        # write_pos 记录当前已经写入了多少个 epoch，用来计算下一个文件的写入区间。
        write_pos = 0
        for info in infos:
            with h5py.File(info.path, "r") as in_file:
                next_pos = write_pos + info.epochs
                out_eeg[write_pos:next_pos, :, :] = in_file["clean_eeg"][...]
                out_labels[write_pos:next_pos] = in_file["labels"][...]
                write_pos = next_pos

    print(f"Done. Wrote: {output_path}")


def main() -> None:
    """脚本入口：从 config 读取路径，并按命令行参数执行拼接。"""

    args = parse_args()
    input_paths = [Path(path) for path in config.PROCESSED_CONCAT_INPUT_FILES]
    output_path = Path(config.PROCESSED_CONCAT_OUTPUT_PATH)
    concatenate_processed_hdf5(
        input_paths=input_paths,
        output_path=output_path,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
