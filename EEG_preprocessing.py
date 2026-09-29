# -*- coding: utf-8 -*-
"""阶段 1/2：HDF5 预处理、连续 Raw 拼接及安全发布。

处理顺序：
    HDF5 数组与元数据读取
    -> MNE RawArray（MNE 内部单位为 V）
    -> 添加任务事件 Annotations
    -> 1 Hz 高通滤波（阶段 1 不做平均参考）
    -> annotate_amplitude 突变检测
    -> 真正的绝对幅值检测
    -> 合并为 bad_preica_amplitude 注释并写回 Raw
    -> mne.Epochs(reject_by_annotation=True) 质量检查

阶段 1 不写 FIF。main() 将本次 Raw 列表交给阶段 2，依赖 MNE 自动
偏移注释，统一 bad_boundary 并保存长 Raw FIF，供独立 ICA 脚本读取。
质量检查 Epochs 不作为拼接输入，最终正式分段仍在 ICA 和最终滤波后。

阶段 2 拼接完成后还会调用共用参考模块：检测持续性坏通道 -> 插值 -> 即时平均参考
-> 记录有效秩 -> 安全发布。不要在每个短 Raw 中分别做这些操作后再拼接。
本文件直接运行时执行 main()；不会自动拟合 ICA，也不会生成阶段 4 分类 HDF5。
建议先看 main() 掌握总流程，再看 preprocess_hdf5_file() 和 concatenate_preprocessed_raws()。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence
import warnings
import xml.etree.ElementTree as ET

import h5py
import mne
import numpy as np
from mne.preprocessing import annotate_amplitude

from eeg_reference import repair_and_reference

import config


# 以下是原采集 HDF5 内部的数据集路径，不是 Windows 文件路径。
SAMPLES_DATASET = "RawData/Samples"  # EEG 数组：原始轴顺序为（采样点，通道）。
ACQUISITION_DATASET = "RawData/AcquisitionTaskDescription"  # XML 字符串：采样率与通道名称。
EVENT_TIME_DATASET = "AsynchronData/Time"  # 事件样本编号，不是以秒为单位的时间。
EVENT_TYPE_DATASET = "AsynchronData/TypeID"  # 事件类型编号，需结合事件定义 XML 解释。
EVENT_VALUE_DATASET = "AsynchronData/Value"  # 事件激活值，按 ACTIVE_EVENT_VALUE 筛选。
EVENT_TYPES_DATASET = "AsynchronData/AsynchronSignalTypes"  # XML：TypeID 到 left/right 等任务名的映射。
TASK_ANNOTATION_REGEXP = r"^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*"  # QC 仅把非 bad/edge 注释当任务事件。


@dataclass(frozen=True)
class HDF5Recording:
    """从一个 HDF5 文件中提取的连续 EEG 和事件信息。"""

    path: Path  # 来源文件，供日志和溯源使用。
    data_volts: np.ndarray  # 已转换为（通道，采样点）的 float64 伏单位数组。
    sampling_rate: float  # 采样率，单位 Hz，来自采集元数据或显式允许的回退值。
    channel_names: tuple[str, ...]  # 与数组第 0 维严格对应，保留采集顺序。
    event_annotations: mne.Annotations  # 任务起点注释，onset 单位秒，duration 为 0。


def read_required_dataset(h5_file: h5py.File, dataset_path: str, source: Path) -> h5py.Dataset:
    """返回必须存在的 HDF5 数据集。"""

    if dataset_path not in h5_file:
        raise KeyError(f"{source}: missing required dataset {dataset_path}")
    dataset = h5_file[dataset_path]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"{source}: {dataset_path} must be an HDF5 dataset")
    return dataset


def read_scalar_text(dataset: h5py.Dataset, dataset_path: str, source: Path) -> str:
    """读取标量或单元素 HDF5 字符串数据集。"""

    value = dataset[()]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{source}: {dataset_path} must contain exactly one string")
        value = value.reshape(-1)[0]

    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="strict")
    if isinstance(value, (str, np.str_)):
        return str(value)
    raise TypeError(
        f"{source}: {dataset_path} has unsupported string type {type(value).__name__}"
    )


def parse_acquisition_metadata(
    acquisition_xml: str,
    channel_count: int,
    source: Path,
) -> tuple[float, tuple[str, ...]]:
    """解析采样率和通道名；配置允许时可使用明确的回退值。"""

    try:
        root = ET.fromstring(acquisition_xml)
    except ET.ParseError as exc:
        if not config.ALLOW_METADATA_FALLBACK:
            raise ValueError(f"{source}: invalid acquisition XML") from exc
        return config.DEFAULT_SFREQ, tuple(f"Ch{index + 1}" for index in range(channel_count))

    sample_rate_text = root.findtext(".//SampleRate")
    channel_names = tuple(
        (element.text or "").strip()
        for element in root.findall(".//ChannelName")
    )

    if sample_rate_text and sample_rate_text.strip():
        sampling_rate = float(sample_rate_text.strip())
    elif config.ALLOW_METADATA_FALLBACK:
        sampling_rate = float(config.DEFAULT_SFREQ)
    else:
        raise ValueError(f"{source}: acquisition XML does not contain SampleRate")

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(f"{source}: invalid sampling rate {sampling_rate!r}")

    valid_names = (
        len(channel_names) == channel_count
        and all(channel_names)
        and len(set(channel_names)) == channel_count
    )
    if not valid_names:
        if not config.ALLOW_METADATA_FALLBACK:
            raise ValueError(
                f"{source}: expected {channel_count} unique non-empty channel names, "
                f"got {channel_names}"
            )
        channel_names = tuple(f"Ch{index + 1}" for index in range(channel_count))

    return sampling_rate, channel_names


def parse_event_type_map(event_types_xml: str, source: Path) -> dict[int, str]:
    """解析 TypeID 到任务事件名称的映射。"""
    # 原采集 TypeID 不等于最终分类标签；先还原任务名称，阶段 4 再固定映射到 1/2。

    try:
        root = ET.fromstring(event_types_xml)
    except ET.ParseError as exc:
        raise ValueError(f"{source}: invalid AsynchronSignalTypes XML") from exc

    type_map: dict[int, str] = {}
    for node in root.findall(".//AsynchronSignalDescription"):
        type_id_text = node.findtext("ID")
        description = (node.findtext("Description") or node.findtext("Name") or "").strip()
        if type_id_text is None or not description:
            continue
        type_id = int(type_id_text)
        if type_id in type_map and type_map[type_id] != description:
            raise ValueError(
                f"{source}: TypeID {type_id} has conflicting descriptions "
                f"{type_map[type_id]!r} and {description!r}"
            )
        type_map[type_id] = description

    if not type_map:
        raise ValueError(f"{source}: no usable event definitions were found")
    return type_map


def convert_samples_to_volts(samples: np.ndarray, source: Path) -> np.ndarray:
    """将形状 (samples, channels) 的源数据转为 (channels, samples) 的 V。"""

    if samples.ndim != 2:
        raise ValueError(f"{source}: {SAMPLES_DATASET} must be 2D, got {samples.shape}")
    if not np.issubdtype(samples.dtype, np.number):
        raise TypeError(f"{source}: EEG samples must be numeric, got {samples.dtype}")
    if not np.isfinite(samples).all():
        raise ValueError(f"{source}: EEG samples contain NaN or infinite values")

    unit = config.SOURCE_DATA_UNIT.strip().lower()
    scale_by_unit = {
        "volts": 1.0,
        "volt": 1.0,
        "v": 1.0,
        "microvolts": 1e-6,
        "microvolt": 1e-6,
        "uv": 1e-6,
        "µv": 1e-6,
    }
    if unit not in scale_by_unit:
        raise ValueError(
            f"Unsupported SOURCE_DATA_UNIT {config.SOURCE_DATA_UNIT!r}; "
            "supported values are volts and microvolts"
        )
    # 转置只改变轴顺序，不改变通道次序；单位在这里转换一次，避免后续重复缩放。
    return np.asarray(samples.T, dtype=np.float64) * scale_by_unit[unit]


def build_event_annotations(
    event_times: np.ndarray,
    event_type_ids: np.ndarray,
    event_values: np.ndarray,
    type_map: dict[int, str],
    sampling_rate: float,
    sample_count: int,
    source: Path,
) -> mne.Annotations:
    """将有效事件的整数样本编号除以采样率，转成相对记录起点的秒单位注释。

    duration=0 仅标出任务触发时刻，不代表运动想象持续 0 秒；试次窗长由后续分段指定。
    """

    times = np.asarray(event_times).reshape(-1)
    type_ids = np.asarray(event_type_ids).reshape(-1)
    values = np.asarray(event_values).reshape(-1)
    if not (len(times) == len(type_ids) == len(values)):
        raise ValueError(
            f"{source}: event length mismatch: Time={len(times)}, "
            f"TypeID={len(type_ids)}, Value={len(values)}"
        )

    onsets: list[float] = []
    descriptions: list[str] = []
    for raw_time, raw_type_id, raw_value in zip(times, type_ids, values):
        if raw_value != config.ACTIVE_EVENT_VALUE:
            continue
        sample_index = int(raw_time)
        if float(raw_time) != sample_index:
            raise ValueError(f"{source}: event time must be an integer sample index, got {raw_time}")
        if sample_index < 0 or sample_index >= sample_count:
            message = (
                f"{source}: event sample {sample_index} is outside "
                f"[0, {sample_count - 1}]"
            )
            if config.SKIP_OUT_OF_RANGE_EVENTS:
                warnings.warn(f"{message}; the event was skipped", RuntimeWarning)
                continue
            raise ValueError(message)

        type_id = int(raw_type_id)
        if type_id not in type_map:
            raise ValueError(f"{source}: TypeID {type_id} is not defined in event metadata")
        description = type_map[type_id]
        if description.lower().startswith("bad_"):
            raise ValueError(
                f"{source}: task event description {description!r} must not start with 'bad_'"
            )
        onsets.append(sample_index / sampling_rate)
        descriptions.append(description)

    if not onsets:
        raise RuntimeError(f"{source}: no active task events were found")
    return mne.Annotations(
        onset=onsets,
        duration=np.zeros(len(onsets), dtype=float),
        description=descriptions,
    )


def read_hdf5_recording(hdf5_path: str | Path) -> HDF5Recording:
    """读取并严格校验一个原始 HDF5 EEG 文件。"""
    # with 块结束即关闭磁盘文件；返回的 HDF5Recording 持有内存数组，不依赖文件句柄。

    source = Path(hdf5_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input HDF5 file not found: {source}")

    with h5py.File(source, "r") as h5_file:
        samples_dataset = read_required_dataset(h5_file, SAMPLES_DATASET, source)
        acquisition_dataset = read_required_dataset(h5_file, ACQUISITION_DATASET, source)
        event_time_dataset = read_required_dataset(h5_file, EVENT_TIME_DATASET, source)
        event_type_dataset = read_required_dataset(h5_file, EVENT_TYPE_DATASET, source)
        event_value_dataset = read_required_dataset(h5_file, EVENT_VALUE_DATASET, source)
        event_types_dataset = read_required_dataset(h5_file, EVENT_TYPES_DATASET, source)

        samples = samples_dataset[...]
        data_volts = convert_samples_to_volts(samples, source)
        acquisition_xml = read_scalar_text(acquisition_dataset, ACQUISITION_DATASET, source)
        sampling_rate, channel_names = parse_acquisition_metadata(
            acquisition_xml,
            channel_count=data_volts.shape[0],
            source=source,
        )
        event_types_xml = read_scalar_text(event_types_dataset, EVENT_TYPES_DATASET, source)
        type_map = parse_event_type_map(event_types_xml, source)
        annotations = build_event_annotations(
            event_times=event_time_dataset[...],
            event_type_ids=event_type_dataset[...],
            event_values=event_value_dataset[...],
            type_map=type_map,
            sampling_rate=sampling_rate,
            sample_count=data_volts.shape[1],
            source=source,
        )

    return HDF5Recording(
        path=source,
        data_volts=data_volts,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
        event_annotations=annotations,
    )


def build_raw(recording: HDF5Recording) -> mne.io.BaseRaw:
    """构建 RawArray 并添加任务事件；明确不执行平均参考。"""

    info = mne.create_info(
        ch_names=list(recording.channel_names),
        sfreq=recording.sampling_rate,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(recording.data_volts, info, verbose=False)
    raw.info["description"] = f"Stage 1 source: {recording.path}"
    raw.set_annotations(recording.event_annotations)
    return raw


def apply_basic_highpass(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """原地应用阶段 1 的 1 Hz IIR 高通滤波。"""
    # h_freq=None 表示不低通；这里修改传入的 Raw，调用方无需接收新的对象。
    # 此时尚未拼接，后续才检测坏段；不能误认为本次高通已按检测出的坏段跳过。

    raw.filter(
        l_freq=config.HIGH_PASS_FREQ,
        h_freq=None,
        picks="eeg",
        method=config.HIGH_PASS_METHOD,
        iir_params={"order": config.IIR_ORDER, "ftype": config.IIR_FILTER_TYPE},
        verbose=False,
    )
    return raw


def _mask_to_intervals(
    mask: np.ndarray,
    sampling_rate: float,
    min_duration: float,
) -> list[tuple[float, float]]:
    """将一维布尔采样掩码转换为 [onset, end) 秒区间。"""

    if mask.ndim != 1:
        raise ValueError(f"Amplitude mask must be 1D, got shape {mask.shape}")
    if not mask.any():
        return []

    # 两端补 False，让开头/结尾的连续 True 也有边界；差分 +1/-1 对应区间起止。
    # stop 是不包含的右端样本号，换成秒后 duration=(stop-start)/sfreq。
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    intervals = []
    for start, stop in zip(starts, stops):
        onset = start / sampling_rate
        end = stop / sampling_rate
        if end - onset + np.finfo(float).eps >= min_duration:
            intervals.append((onset, end))
    return intervals


def amplitude_settings(post_ica=False):
    """两阶段共用检测参数快照；ICA 后参数独立，名称固定以防误删永久坏段。"""
    if (config.PREICA_BAD_AMPLITUDE_LABEL != "bad_preica_amplitude"
            or config.BAD_AMPLITUDE_LABEL != "bad_amplitude"):
        raise ValueError("Keep the pre/post ICA amplitude annotation names unchanged")
    if post_ica:
        settings = dict(
            peak_uv=config.POSTICA_AMPLITUDE_PEAK_UV,
            peak_min_duration=config.POSTICA_AMPLITUDE_PEAK_MIN_DURATION,
            absolute_uv=config.POSTICA_ABSOLUTE_AMPLITUDE_THRESHOLD_UV,
            absolute_min_duration=config.POSTICA_ABSOLUTE_AMPLITUDE_MIN_DURATION,
            merge_gap=config.POSTICA_AMPLITUDE_MERGE_GAP_SECONDS,
            bad_percent=100.0, label=config.BAD_AMPLITUDE_LABEL,
        )
    else:
        settings = dict(
            peak_uv=config.ANNOTATE_AMPLITUDE_PEAK_UV,
            peak_min_duration=config.ANNOTATE_AMPLITUDE_MIN_DURATION,
            absolute_uv=config.ABSOLUTE_AMPLITUDE_THRESHOLD_UV,
            absolute_min_duration=config.ABSOLUTE_AMPLITUDE_MIN_DURATION,
            merge_gap=config.BAD_AMPLITUDE_MERGE_GAP_SECONDS,
            bad_percent=config.ANNOTATE_AMPLITUDE_BAD_PERCENT,
            label=config.PREICA_BAD_AMPLITUDE_LABEL,
        )
    for key in ("peak_uv", "peak_min_duration", "absolute_uv", "absolute_min_duration", "merge_gap"):
        value = settings[key]
        if (type(value) not in (int, float) or not np.isfinite(value)
                or value < 0 or (key != "merge_gap" and value == 0)):
            raise ValueError(f"Amplitude {key} must be finite and positive (merge_gap may be zero)")
    if settings["bad_percent"] != 100:
        raise ValueError("Keep amplitude bad_percent at 100 for time annotations")
    return settings


def detect_absolute_amplitude_intervals(raw: mne.io.BaseRaw, *, settings=None) -> list[tuple[float, float]]:
    """检测任一 EEG 通道绝对幅值超阈值的连续时间段。"""

    settings = amplitude_settings() if settings is None else settings
    threshold_volts = settings["absolute_uv"] * 1e-6
    eeg_data = raw.get_data(picks="eeg")
    # axis=0 汇总所有通道：只要一个电极超限，该时间点就进入全局坏段掩码。
    amplitude_mask = np.any(np.abs(eeg_data) > threshold_volts, axis=0)
    return _mask_to_intervals(
        amplitude_mask,
        sampling_rate=float(raw.info["sfreq"]),
        min_duration=settings["absolute_min_duration"],
    )


def detect_peak_change_intervals(raw: mne.io.BaseRaw, *, settings=None) -> list[tuple[float, float]]:
    """调用 annotate_amplitude 检测相邻采样点的异常突变时间段。"""
    # peak 检查的是相邻点差值；flat=None 在此关闭平线检测，持续平线由阶段 2 坏通道模块处理。

    settings = amplitude_settings() if settings is None else settings
    annotations, bad_channels = annotate_amplitude(
        raw,
        peak={"eeg": settings["peak_uv"] * 1e-6},
        flat=None,
        bad_percent=settings["bad_percent"],
        min_duration=settings["peak_min_duration"],
        picks="eeg",
        verbose=False,
    )
    if bad_channels:
        raise RuntimeError(
            "annotate_amplitude returned bad channels instead of time annotations: "
            f"{bad_channels}. Keep ANNOTATE_AMPLITUDE_BAD_PERCENT at 100 for this stage."
        )

    intervals = []
    sample_duration = 1.0 / float(raw.info["sfreq"])
    for onset, duration in zip(annotations.onset, annotations.duration):
        intervals.append((float(onset), float(onset + max(duration, sample_duration))))
    return intervals


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
    max_gap_seconds: float = 0.0,
) -> list[tuple[float, float]]:
    """合并重叠或间隔不超过指定长度的异常区间。"""

    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + max_gap_seconds:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def annotate_bad_amplitude(raw: mne.io.BaseRaw, *, settings=None) -> mne.Annotations:
    """合并两种幅值检测结果，统一命名并追加到 Raw。"""

    settings = amplitude_settings() if settings is None else settings
    if raw.first_samp != 0 or raw.annotations.orig_time is not None:
        raise ValueError("Amplitude detection requires relative-time Raw with first_samp=0")
    peak_intervals = detect_peak_change_intervals(raw, settings=settings)
    absolute_intervals = detect_absolute_amplitude_intervals(raw, settings=settings)
    merged = merge_intervals(
        [*peak_intervals, *absolute_intervals],
        max_gap_seconds=settings["merge_gap"],
    )
    bad_annotations = mne.Annotations(
        onset=[start for start, _ in merged],
        duration=[end - start for start, end in merged],
        description=[settings["label"]] * len(merged),
    )
    # 只增加注释，不删样本、不改电压，也不丢掉已有 left/right 任务事件。
    raw.set_annotations(raw.annotations + bad_annotations)
    return bad_annotations


def create_epochs_from_raw(
    raw: mne.io.BaseRaw,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    baseline: tuple[float | None, float | None] | None = None,
) -> mne.Epochs:
    """创建临时 QC Epochs，拒绝 bad* 重叠或越界试次，不改变连续 Raw。

    tmin/tmax/baseline 为 None 时使用对应 QC 配置；这里的 baseline=None 是“使用配置”，
    而非强制关闭校正（需将 QC_EPOCH_BASELINE 设为 None）。
    事件编码由 MNE 自动生成，仅用于 QC；阶段 4 使用独立的固定 left/right 映射。
    """

    events, event_id = mne.events_from_annotations(
        raw,
        regexp=TASK_ANNOTATION_REGEXP,
        verbose=False,
    )
    if events.size == 0:
        raise RuntimeError("No task events are available for epoch creation")

    epoch_tmin = config.QC_EPOCH_TMIN if tmin is None else tmin
    epoch_tmax = config.QC_EPOCH_TMAX if tmax is None else tmax
    epoch_baseline = config.QC_EPOCH_BASELINE if baseline is None else baseline
    return mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=epoch_tmin,
        tmax=epoch_tmax,
        baseline=epoch_baseline,
        preload=True,
        reject_by_annotation=True,
        verbose=False,
    )


def preprocess_hdf5_file(hdf5_path: str | Path) -> mne.io.BaseRaw:
    """执行单个 HDF5 文件的阶段 1 预处理。"""
    # 返回仍包含坏段和完整时间轴的连续 Raw；此函数不插值、不平均参考、不写 FIF。

    recording = read_hdf5_recording(hdf5_path)
    raw = build_raw(recording)
    apply_basic_highpass(raw)
    bad_annotations = annotate_bad_amplitude(raw)
    print(
        f"Preprocessed {recording.path.name}: channels={len(raw.ch_names)}, "
        f"samples={raw.n_times}, events={len(recording.event_annotations)}, "
        f"bad_amplitude_segments={len(bad_annotations)}"
    )
    return raw


def preprocess_hdf5_files(
    input_paths: Sequence[str | Path] | None = None,
    *,
    run_epoch_check: bool | None = None,
) -> list[mne.io.BaseRaw]:
    """批量预处理并返回连续 Raw 列表，作为阶段 2 的直接输入。"""

    paths = list(input_paths) if input_paths is not None else config.get_input_hdf5_paths()
    if not paths:
        raise ValueError("At least one HDF5 input path is required")

    should_check_epochs = (
        config.RUN_EPOCH_REJECTION_CHECK if run_epoch_check is None else run_epoch_check
    )
    raw_list: list[mne.io.BaseRaw] = []
    for path in paths:
        raw = preprocess_hdf5_file(path)
        if should_check_epochs:
            # 临时 Epoch 只用来打印保留/拒绝数量；追加到 raw_list 的始终是原连续 Raw。
            epochs = create_epochs_from_raw(raw)
            rejected = len(epochs.drop_log) - len(epochs)
            print(
                f"  Epoch annotation check: accepted={len(epochs)}, "
                f"rejected_or_out_of_bounds={rejected}"
            )
        raw_list.append(raw)
    return raw_list


def create_epochs_list(raw_list: Sequence[mne.io.BaseRaw]) -> list[mne.Epochs]:
    """显式批量生成 QC Epochs 的便捷接口；当前正式阶段 4 使用独立的分段函数。"""

    return [create_epochs_from_raw(raw) for raw in raw_list]


def _metadata_equal(left, right) -> bool:
    """比较包含 ndarray 的 MNE 元数据，不进行静默修正。"""

    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _metadata_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _metadata_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right, equal_nan=True)
    return left == right


def _validate_annotations(raw: mne.io.BaseRaw) -> None:
    """仅检查注释的基本有效范围，不计算或验证全局偏移。"""

    annotations = raw.annotations
    end = raw.n_times / raw.info["sfreq"]
    tolerance = 1.0 / raw.info["sfreq"]
    if (
        not np.isfinite(annotations.onset).all()
        or not np.isfinite(annotations.duration).all()
        or np.any(annotations.duration < 0)
        or np.any(annotations.onset < -tolerance)
        or np.any(annotations.onset + annotations.duration > end + tolerance)
    ):
        raise ValueError("Annotations contain invalid times or exceed recording bounds")


def validate_raw_list(raw_list: Sequence[mne.io.BaseRaw]) -> None:
    """校验阶段 1 的短 Raw；不接受已裁剪、已拼接或绝对时间输入。"""

    if not raw_list:
        raise ValueError("Raw list must not be empty")
    if len({id(raw) for raw in raw_list}) != len(raw_list):
        raise ValueError("Raw list contains duplicate object references")
    for index, raw in enumerate(raw_list):
        prefix = f"Raw[{index}]"
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError(f"{prefix}: expected MNE BaseRaw")
        if not raw.preload or raw.n_times <= 0 or not raw.ch_names:
            raise ValueError(f"{prefix}: expected non-empty preloaded data")
        if raw.first_samp != 0 or raw.info["meas_date"] is not None or raw.annotations.orig_time is not None:
            raise ValueError(f"{prefix}: expected stage 1 relative time with first_samp=0")
        sfreq = raw.info["sfreq"]
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError(f"{prefix}: invalid sampling rate")
        if raw.info["highpass"] != config.HIGH_PASS_FREQ:
            raise ValueError(f"{prefix}: highpass differs from stage 1 configuration")
        if len(set(raw.ch_names)) != len(raw.ch_names) or set(raw.get_channel_types()) != {"eeg"}:
            raise ValueError(f"{prefix}: expected unique EEG channels")
        if raw.info["custom_ref_applied"] or raw.info["projs"]:
            raise ValueError(f"{prefix}: stage 1 must not apply a new reference or projections")
        if not set(raw.info["bads"]).issubset(raw.ch_names):
            raise ValueError(f"{prefix}: unknown bad channel names")
        for channel in raw.info["chs"]:
            if channel["unit"] != mne.io.constants.FIFF.FIFF_UNIT_V:
                raise ValueError(f"{prefix}: EEG channel unit must be volts")
        # 分块检查，避免校验时复制整个长记录。
        for start in range(0, raw.n_times, 100000):
            data = raw.get_data(start=start, stop=min(start + 100000, raw.n_times))
            if not np.isrealobj(data) or not np.isfinite(data).all():
                raise ValueError(f"{prefix}: data must be real and finite")
        _validate_annotations(raw)
        if any("boundary" in str(label).lower() for label in raw.annotations.description):
            raise ValueError(f"{prefix}: already contains boundary annotations")
        if index:
            # 同名但不同顺序的通道也不兼容；不自动排序/重采样/修正元数据来掩盖输入问题。
            first = raw_list[0]
            for key in (
                "sfreq", "ch_names", "nchan", "highpass", "lowpass", "bads",
                "projs", "custom_ref_applied", "dev_head_t", "dig", "chs", "line_freq",
            ):
                if not _metadata_equal(first.info.get(key), raw.info.get(key)):
                    raise ValueError(f"{prefix}: incompatible {key} compared with Raw[0]")


def save_concatenated_raw(raw: mne.io.BaseRaw, output_path: str | Path | None = None) -> Path:
    """先保存及回读单文件 FIF，再原子替换；失败时保留上一次有效文件。

    不支持分卷发布：不能把含临时路径引用的分卷主文件直接重命名。
    独立 ICA 脚本应在本函数成功结束后重新加载正式路径。
    """

    output = Path(output_path if output_path is not None else config.CONCATENATED_RAW_PATH).resolve()
    if not output.name.endswith("_raw.fif"):
        raise ValueError("Stage 2 output filename must end with _raw.fif")
    if config.CONCATENATED_RAW_FORMAT not in {"single", "double"}:
        raise ValueError("FIF format must be single or double, not an integer format")
    if output.exists() and not config.CONCATENATED_RAW_OVERWRITE:
        raise FileExistsError(f"Stage 2 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="stage2-", dir=output.parent) as temporary_dir:
        temporary_path = Path(temporary_dir) / "stage2_raw.fif"
        raw.save(temporary_path, fmt=config.CONCATENATED_RAW_FORMAT, overwrite=False, verbose=False)
        if len(list(Path(temporary_dir).glob("*.fif"))) != 1:
            raise RuntimeError("Stage 2 output requires split FIF files; previous output was retained")
        with mne.io.read_raw_fif(temporary_path, preload=False, verbose=False) as restored:
            if restored.n_times != raw.n_times or restored.ch_names != raw.ch_names:
                raise RuntimeError("Saved FIF shape or channel names differ from the long Raw")
            if restored.info["sfreq"] != raw.info["sfreq"]:
                raise RuntimeError("Saved FIF sampling rate mismatch")
            if Counter(restored.annotations.description) != Counter(raw.annotations.description):
                raise RuntimeError("Saved FIF annotation counts differ from the long Raw")
            _validate_annotations(restored)
            # 回读全部样本，验证文件完整性和保存精度；不逐项验证注释时间。
            for start in range(0, raw.n_times, 100000):
                stop = min(start + 100000, raw.n_times)
                if not np.allclose(
                    restored.get_data(start=start, stop=stop),
                    raw.get_data(start=start, stop=stop), rtol=1e-6, atol=1e-12,
                ):
                    raise RuntimeError("Saved FIF data differs from the long Raw")
        if output.exists() and not config.CONCATENATED_RAW_OVERWRITE:
            raise FileExistsError(f"Stage 2 output already exists: {output}")
        os.replace(temporary_path, output)
    print(f"Published latest stage 2 Raw: {output}")
    return output


def concatenate_preprocessed_raws(
    raw_list: Sequence[mne.io.BaseRaw],
    *,
    save_output: bool = True,
    output_path: str | Path | None = None,
) -> mne.io.BaseRaw:
    """校验并拼接副本，所有事件时间偏移由 MNE 自动完成。"""

    validate_raw_list(raw_list)
    if config.CONCATENATION_ON_MISMATCH != "raise":
        raise ValueError("CONCATENATION_ON_MISMATCH must remain 'raise'")
    if config.BAD_BOUNDARY_LABEL != "bad_boundary":
        raise ValueError("BAD_BOUNDARY_LABEL must be 'bad_boundary'")
    counts = Counter(label for raw in raw_list for label in raw.annotations.description)
    total_samples = sum(raw.n_times for raw in raw_list)
    # concatenate_raws 会修改列表里的首个对象，因此传入副本，避免污染调用者的短 Raw。
    # 事件全局时间偏移由 MNE 完成，不能再手动加一次各文件长度。
    combined = mne.concatenate_raws(
        [raw.copy() for raw in raw_list], preload=True,
        on_mismatch=config.CONCATENATION_ON_MISMATCH, verbose=False,
    )
    # 仅规范描述，不计算、重建或改动任何 Annotation 时间。
    combined.annotations.delete(np.flatnonzero(combined.annotations.description == "EDGE boundary"))
    if "BAD boundary" in combined.annotations.description:
        combined.annotations.rename({"BAD boundary": config.BAD_BOUNDARY_LABEL})
    if len(raw_list) > 1:
        counts[config.BAD_BOUNDARY_LABEL] = len(raw_list) - 1
    if combined.n_times != total_samples or Counter(combined.annotations.description) != counts:
        raise RuntimeError("Concatenation sample or annotation count mismatch")
    _validate_annotations(combined)
    # 全部短段已拼好后，统一检测/修复坏通道并平均参考；完整时间轴与事件注释不变。
    combined, reference_record = repair_and_reference(combined)
    # FIF 的 info description 有长度限制；这里只写入 ICA 必需的紧凑契约，
    # 检测分数等完整信息仍由调用方可从 reference_record 获取/另存。
    compact_reference = {
        key: reference_record[key]
        for key in ("schema_version", "reference", "projection", "channel_names",
                    "n_times", "montage", "interpolated_channels", "eeg_rank",
                    "rank_after_interpolation", "mne_rank_auto", "rank_rel_tol")
    }
    combined.info["description"] = json.dumps({
        "stage": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [raw.info.get("description") for raw in raw_list],
        "eeg_reference": compact_reference,
    }, ensure_ascii=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
    if save_output:
        save_concatenated_raw(combined, output_path)
    print(f"Stage 2 complete: samples={combined.n_times}, boundaries={len(raw_list) - 1}")
    return combined


def main() -> mne.io.BaseRaw:
    """串联本次阶段 1 和阶段 2，发布并返回最新长 Raw。"""
    # 文件顺序取自配置；阶段 3/4 必须由各自入口另行运行，尤其不能自动跳过 ICA 人工审核。

    input_paths = config.get_input_hdf5_paths()
    print("Stage 1 input files:")
    for path in input_paths:
        print(f"  - {path}")
    raw_list = preprocess_hdf5_files(input_paths)
    print(f"Stage 1 complete. Raw objects ready for concatenation: {len(raw_list)}")
    return concatenate_preprocessed_raws(raw_list)


if __name__ == "__main__":
    main()