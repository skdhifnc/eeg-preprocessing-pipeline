# -*- coding: utf-8 -*-
"""阶段 4：将人工 ICA 审核后的连续 EEG 按配置导出为试次数据。

输入：正式阶段 3 clean Raw FIF，以及用来验证来源的模型清单和 apply 凭据。
流程：校验来源 -> 按配置对连续 Raw 带通 -> 按配置事件和完整窗口分段
      -> 拒绝坏段/越界 Epoch -> 裁出配置的分类窗口 -> 发布两份 HDF5 和报告。
输出形状为（试次数，通道数，时间点数）；时间窗、单位、dtype 和标签映射均来自 config.py。

Raw 是连续记录；Epochs 是以任务事件为零时刻的一组试次，二者不可混作同一种输入。
本阶段不重做 ICA、插值或参考，也不做分类训练。
直接运行本文件会按配置发布输出；仅 import 时不会执行 main()。
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import warnings

import h5py
import mne
import numpy as np

import config

from ICA_processing import file_hash, validate_exclusions
from EEG_preprocessing import amplitude_settings
from eeg_reference import check_average_reference
from processed_data_concatenation import inspect_processed_file
from ws import validate_clean_hdf5

OUTPUT_UNIT_SCALES = {
    "volts": 1.0,
    "millivolts": 1e3,
    "microvolts": 1e6,
    "nanovolts": 1e9,
}


def utc_now():
    """生成带时区的 UTC 时间字符串，供报告记录本次处理时间。"""
    return datetime.now(timezone.utc).isoformat()


def validate_config(*, validate_outputs=True):
    """验证配置之间的逻辑关系和路径安全，不限制具体导出规格。"""
    if Path(config.STAGE4_INPUT_RAW_PATH).resolve() != Path(config.ICA_CLEAN_RAW_PATH).resolve():
        raise ValueError("Stage 4 only accepts ICA_CLEAN_RAW_PATH, never a stage 2 fallback")
    if type(config.STAGE4_OVERWRITE) is not bool:
        raise ValueError("STAGE4_OVERWRITE must be a bool")

    event_id = config.STAGE4_EVENT_ID
    if not isinstance(event_id, dict) or len(event_id) != 2:
        raise ValueError("STAGE4_EVENT_ID must define exactly two classes")
    if any(not isinstance(label, str) or not label for label in event_id):
        raise ValueError("STAGE4_EVENT_ID class names must be non-empty strings")
    if (any(type(value) is not int for value in event_id.values())
            or len(set(event_id.values())) != 2):
        raise ValueError("STAGE4_EVENT_ID values must be two distinct integers")

    numeric_values = {
        "STAGE4_FILTER_L_FREQ": config.STAGE4_FILTER_L_FREQ,
        "STAGE4_FILTER_H_FREQ": config.STAGE4_FILTER_H_FREQ,
        "STAGE4_EPOCH_TMIN": config.STAGE4_EPOCH_TMIN,
        "STAGE4_EPOCH_TMAX": config.STAGE4_EPOCH_TMAX,
        "STAGE4_WS_TMIN": config.STAGE4_WS_TMIN,
        "STAGE4_WS_TMAX": config.STAGE4_WS_TMAX,
    }
    for key, value in numeric_values.items():
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if not (0 < config.STAGE4_FILTER_L_FREQ < config.STAGE4_FILTER_H_FREQ):
        raise ValueError("Invalid stage 4 band-pass frequencies")
    if config.STAGE4_EPOCH_TMIN >= config.STAGE4_EPOCH_TMAX:
        raise ValueError("STAGE4_EPOCH_TMIN must be less than STAGE4_EPOCH_TMAX")
    if not (config.STAGE4_EPOCH_TMIN <= config.STAGE4_WS_TMIN < config.STAGE4_WS_TMAX
            <= config.STAGE4_EPOCH_TMAX):
        raise ValueError("WS crop must be inside the full epoch")

    baseline = config.STAGE4_BASELINE
    if baseline is not None:
        if not isinstance(baseline, (tuple, list)) or len(baseline) != 2:
            raise ValueError("STAGE4_BASELINE must be None or a two-item interval")
        start = config.STAGE4_EPOCH_TMIN if baseline[0] is None else baseline[0]
        stop = config.STAGE4_EPOCH_TMAX if baseline[1] is None else baseline[1]
        if (type(start) not in (int, float) or type(stop) not in (int, float)
            or not np.isfinite(start) or not np.isfinite(stop)
            or not (config.STAGE4_EPOCH_TMIN <= start <= stop <= config.STAGE4_EPOCH_TMAX)):
            raise ValueError("STAGE4_BASELINE must be ordered and inside the full epoch")

    if config.STAGE4_OUTPUT_UNIT not in OUTPUT_UNIT_SCALES:
        raise ValueError(f"Unsupported STAGE4_OUTPUT_UNIT: {config.STAGE4_OUTPUT_UNIT!r}")
    try:
        output_dtype = np.dtype(config.STAGE4_OUTPUT_DTYPE)
    except TypeError as exc:
        raise ValueError("Invalid STAGE4_OUTPUT_DTYPE") from exc
    if not np.issubdtype(output_dtype, np.floating):
        raise ValueError("STAGE4_OUTPUT_DTYPE must be a floating-point dtype")

    skip = config.STAGE4_FILTER_SKIP_ANNOTATIONS
    if not isinstance(skip, (tuple, list)) or any(
        not isinstance(name, str) or not name for name in skip):
        raise ValueError("STAGE4_FILTER_SKIP_ANNOTATIONS must be a sequence of names")
    required_skip = {"bad", "edge", "bad_acq_skip", config.BAD_BOUNDARY_LABEL,
                 config.BAD_AMPLITUDE_LABEL}
    if not required_skip.issubset(skip):
        raise ValueError("Do not remove stage 4 filter boundary/bad annotation exclusions")

    if not validate_outputs:
        return  # FBCSP 复用处理设置校验，但有独立的输出契约和覆盖保护。
    outputs = output_paths()
    protected = {Path(p).resolve() for p in (
        config.STAGE4_INPUT_RAW_PATH, config.ICA_INPUT_RAW_PATH, config.ICA_MODEL_PATH,
        config.ICA_MANIFEST_PATH, config.ICA_CLEAN_RAW_PATH)}
    if len(set(outputs)) != 3 or protected.intersection(outputs):
        raise ValueError("Stage 4 outputs must be distinct and must not overwrite upstream files")
    if any(path.suffix != ".hdf5" for path in outputs[:2]) or outputs[2].suffix != ".json":
        raise ValueError("Expected two .hdf5 outputs and one .json report")
    if not config.STAGE4_OVERWRITE and any(path.exists() for path in outputs):
        raise FileExistsError("Stage 4 output already exists")


def output_paths():
    """按固定顺序返回绝对路径：[完整 HDF5，ws HDF5，报告]，报告最后发布。"""
    return [Path(p).resolve() for p in (config.STAGE4_FULL_OUTPUT_PATH,
                                      config.STAGE4_WS_OUTPUT_PATH, config.STAGE4_REPORT_PATH)]


def read_json(path):
    """读取 UTF-8 JSON 记录；缺失或损坏时由异常阻止继续使用旧结果。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_source(source):
    """发布前再次检查输入、模型、manifest 和 apply 凭据，拒绝过期 clean。"""
    for path, expected in source["dependencies"].items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Stage 3 dependency changed: {path}; rerun fit/review/apply")


def load_clean_raw():
    """返回 (连续 Raw 副本, 来源记录 source)，不读取阶段 2 EEG 作为替代输入。

    除了检查数据形状，还检查参考、注释、FIF 内嵌记录、外部 apply 凭据是否一致。
    source.dependencies 保存所有依赖文件的 SHA-256，发布前再验一次，防止中途换文件。
    """
    path = Path(config.STAGE4_INPUT_RAW_PATH).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 4 requires the published stage 3 clean Raw: {path}; run ICA apply first"
        )
    # 同名文件可能已被重新发布，因此比较内容哈希，而不是只看路径/修改时间。
    before = file_hash(path)
    with mne.io.read_raw_fif(path, preload=True, verbose=False) as loaded:
        if len(loaded.filenames) != 1:
            raise ValueError("Stage 4 supports only a single-file stage 3 Raw")
        raw = loaded.copy()
    if before != file_hash(path):
        raise RuntimeError("Stage 3 input changed while loading")
    check_average_reference(raw)
    if raw.first_samp != 0 or raw.info["meas_date"] is not None or raw.annotations.orig_time is not None:
        raise ValueError("Expected stage 3 relative-time Raw with first_samp=0")
    if len(raw.ch_names) != 32 or set(raw.get_channel_types()) != {"eeg"}:
        raise ValueError("Stage 4 requires 32 EEG channels")
    if config.STAGE4_FILTER_H_FREQ >= raw.info["sfreq"] / 2:
        raise ValueError("STAGE4_FILTER_H_FREQ must be below the input Nyquist frequency")
    if set(raw.ch_names) != set(config.ICA_DEVICE_CHANNELS) or not raw.info["dig"]:
        raise ValueError("Stage 4 requires standard device channels and electrode coordinates")
    if raw.info["highpass"] < 1:
        raise ValueError("Expected stage 1 highpass before ICA")
    if not np.isfinite(raw.get_data()).all():
        raise ValueError("Stage 3 Raw contains non-finite data")
    descriptions = set(map(str, raw.annotations.description))
    unknown = {name for name in descriptions if name not in config.STAGE4_EVENT_ID
               and not name.lower().startswith(("bad", "edge"))}
    if unknown:
        raise ValueError(f"Unknown task annotations: {sorted(unknown)}")
    for label in config.STAGE4_EVENT_ID:
        if label not in descriptions:
            raise ValueError(f"Missing required task annotation: {label}")
    # manifest 描述“模型来自哪里”；receipt 描述“该模型用哪些排除成分生成了哪个 clean”。
    manifest_path = Path(config.ICA_MANIFEST_PATH).resolve()
    manifest_hash = file_hash(manifest_path)
    manifest = read_json(manifest_path)
    try:
        applied = json.loads(raw.info.get("description") or "{}")
        model_id = manifest["model_id"]
        if not isinstance(model_id, str) or len(model_id) != 32 or any(
                char not in "0123456789abcdef" for char in model_id):
            raise ValueError("Invalid stage 3 model ID")
        receipt_path = Path(config.ICA_REPORT_DIR) / model_id / "applied_components.json"
        receipt_hash = file_hash(receipt_path)
        receipt = read_json(receipt_path)
        for key in ("model_id", "model_sha256", "input_sha256", "excluded_components", "applied_at", "input_path",
                    "postica_amplitude_settings", "amplitude_recheck"):
            if key not in applied or applied[key] != receipt[key]:
                raise ValueError(f"Stage 3 apply receipt mismatch: {key}")
        if (applied["model_id"] != model_id or applied["model_sha256"] != manifest["model_sha256"]
                or applied["input_sha256"] != manifest["source"]["sha256"]
                or receipt["output_sha256"] != before
                or manifest["source"]["ch_names"] != raw.ch_names
                or manifest["source"]["n_times"] != raw.n_times
                or manifest["source"]["sfreq"] != raw.info["sfreq"]):
            raise ValueError("Stage 3 clean/manifest/receipt mismatch; run ICA apply again")
        if Path(applied["input_path"]).resolve() != Path(config.ICA_INPUT_RAW_PATH).resolve():
            raise ValueError("Stage 3 receipt uses a different stage 2 input")
        configured_exclusions = validate_exclusions(config.ICA_EXCLUDE_COMPONENTS, manifest["n_components"])
        if applied["excluded_components"] != configured_exclusions:
            raise ValueError("ICA selection changed since publication; preview/apply again")
        if (applied["postica_amplitude_settings"] != amplitude_settings(post_ica=True)
                or applied["amplitude_recheck"].get("schema_version") != 1
                or applied["amplitude_recheck"].get("settings") != applied["postica_amplitude_settings"]):
            raise ValueError("Post-ICA amplitude settings changed or recheck missing; preview/apply again")
        if config.PREICA_BAD_AMPLITUDE_LABEL in raw.annotations.description:
            raise ValueError("Pre-ICA amplitude annotations remain; run preview/apply again")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Missing/invalid stage 3 publication record; run ICA apply first") from exc
    source = dict(path=str(path), sha256=before, applied=applied, dependencies={
        str(path): before, str(manifest_path): manifest_hash,
        str(receipt_path.resolve()): receipt_hash,
        str(Path(config.ICA_MODEL_PATH).resolve()): manifest["model_sha256"],
        str(Path(config.ICA_INPUT_RAW_PATH).resolve()): manifest["source"]["sha256"],
    })
    verify_source(source)
    return raw, source


def filter_raw(raw):
    """返回带通后的 Raw 副本；保留输入的全部样本、通道顺序和注释。

    在连续数据上先滤波，避免每个 Epoch 单独滤波造成重复的窗边缘效应。
    skip_by_annotation 将坏段和拼接边界隔开；被跳过的坏段仍保留，稍后拒绝受影响试次。
    """
    filtered = raw.copy()
    filtered.filter(
        l_freq=config.STAGE4_FILTER_L_FREQ,
        h_freq=config.STAGE4_FILTER_H_FREQ,
        picks="eeg", method=config.STAGE4_FILTER_METHOD,
        phase=config.STAGE4_FILTER_PHASE, fir_design=config.STAGE4_FILTER_DESIGN,
        skip_by_annotation=config.STAGE4_FILTER_SKIP_ANNOTATIONS,
        verbose=False,
    )
    if filtered.ch_names != raw.ch_names or filtered.n_times != raw.n_times:
        raise RuntimeError("Stage 4 filtering changed Raw shape")
    for key in ("onset", "duration", "description"):
        np.testing.assert_array_equal(getattr(filtered.annotations, key), getattr(raw.annotations, key))
    if not np.isfinite(filtered.get_data()).all():
        raise RuntimeError("Stage 4 filtering produced non-finite data")
    check_average_reference(filtered)
    return filtered


def make_epochs(filtered):
    """按配置的事件编码和时间范围切完整审核窗口，返回通过坏段拒绝的预加载 Epochs。"""
    # MNE 只自动拒绝 bad*；edge* 在分段副本中增加别名，原 Raw 注释不变。
    filtered = filtered.copy()
    for annotation in list(filtered.annotations):
        if annotation["description"].lower().startswith("edge"):
            filtered.annotations.append(annotation["onset"], annotation["duration"], "bad_permanent_edge")
    events, _ = mne.events_from_annotations(
        filtered, event_id=config.STAGE4_EVENT_ID, regexp=None, verbose=False
    )
    if len(events) == 0:
        raise RuntimeError("No configured task events available")
    # events 每行是 [原 Raw 样本编号, 前值, 事件编码]；第 3 列才是配置后的分类标签。
    # reject_by_annotation 拒绝 bad* 重叠；重复事件直接报错，不静默合并试次。
    epochs = mne.Epochs(
        filtered, events=events, event_id=config.STAGE4_EVENT_ID,
        tmin=config.STAGE4_EPOCH_TMIN, tmax=config.STAGE4_EPOCH_TMAX,
        baseline=config.STAGE4_BASELINE, picks="eeg", preload=True,
        reject_by_annotation=True, event_repeated="error", verbose=False,
    )
    if len(epochs) == 0:
        raise RuntimeError("All stage 4 epochs were rejected")
    counts = Counter(epochs.events[:, 2].tolist())
    missing = [label for label in config.STAGE4_EVENT_ID.values() if not counts.get(label)]
    if missing:
        raise RuntimeError(f"One task class disappeared after rejection: {counts}")
    return epochs


def rejection_report(epochs, raw):
    """把 MNE 的拒绝日志关联回原始任务事件，返回可写入 JSON 的统计字典。

    drop_log 对应原事件序列，空元组表示保留；selection 是保留试次的原事件索引。
    MNE 返回的原因不保证穷举同一试次重叠的所有坏段，不能当成完整坏段清单。
    """
    events, _ = mne.events_from_annotations(raw, event_id=config.STAGE4_EVENT_ID,
                                          regexp=None, verbose=False)
    reasons = Counter(reason for log in epochs.drop_log for reason in log)
    inverse = {v: k for k, v in config.STAGE4_EVENT_ID.items()}
    rejected = [dict(event_index=i, event_sample=int(event[0]),
                     onset_seconds=float(event[0] / raw.info["sfreq"]),
                     label=inverse[int(event[2])], reasons=list(epochs.drop_log[i]))
                for i, event in enumerate(events) if epochs.drop_log[i]]
    bad_rejected = sum(
        any(str(reason).lower().startswith(("bad", "edge")) for reason in item["reasons"])
        for item in rejected
    )
    categories = {"pre_ica": 0, "post_ica": 0, "permanent": 0}
    for item in rejected:
        names = item["reasons"]
        categories["pre_ica"] += int(config.PREICA_BAD_AMPLITUDE_LABEL in names)
        categories["post_ica"] += int(config.BAD_AMPLITUDE_LABEL in names)
        categories["permanent"] += int(any(
            name.lower().startswith(("bad", "edge"))
            and name not in (config.PREICA_BAD_AMPLITUDE_LABEL, config.BAD_AMPLITUDE_LABEL) for name in names))
    return {
        "total_events": len(epochs.drop_log),
        "accepted_epochs": len(epochs),
        "rejected_epochs": sum(bool(log) for log in epochs.drop_log),
        "bad_annotation_rejected_epochs": bad_rejected,
        "bad_annotation_rejected_by_category": categories,
        "rejection_reasons": dict(reasons),
        "reason_semantics": "MNE drop_log reasons, not an exhaustive list of overlapping annotations",
        "rejected_events": rejected,
        "original_labels": {str(k): int(v) for k, v in Counter(events[:, 2]).items()},
        "kept_event_indices": epochs.selection.tolist(),
        "accepted_labels": {str(k): int(v) for k, v in Counter(epochs.events[:, 2]).items()},
    }


def print_epoch_rejection_summary(full_epochs, ws_epochs, rejection):
    """在终端打印阶段 4 切分、拒绝和分类窗口裁剪统计。"""
    print("Stage 4 Epoch 切分统计")
    print(f"  完整审核窗口: {full_epochs.times[0]:g} ~ {full_epochs.times[-1]:g} s")
    print(f"  候选 Epoch 总数: {rejection['total_events']}")
    print(f"  丢弃 Epoch 总数: {rejection['rejected_epochs']}")
    print(f"  其中因 bad*/edge 坏段注释丢弃（按 Epoch 去重）: {rejection['bad_annotation_rejected_epochs']}")
    print(f"  最终保留 Epoch 数: {rejection['accepted_epochs']}")
    print(f"  坏段原因分类（MNE 实际返回原因，非所有重叠区间）: {rejection['bad_annotation_rejected_by_category']}")
    print("  拒绝原因计数（同一 Epoch 可能同时命中多个原因）:")
    if rejection["rejection_reasons"]:
        for reason, count in sorted(rejection["rejection_reasons"].items()):
            print(f"    - {reason}: {count}")
    else:
        print("    - 无")
    print(f"  分类窗口: {ws_epochs.times[0]:g} ~ {ws_epochs.times[-1]:g} s；Epoch 数: {len(ws_epochs)}（仅裁剪时间，不额外丢弃）")


def arrays(epochs):
    """按配置的单位和浮点类型生成 HDF5 数组，不改写 Epochs。

    返回顺序：EEG、标签(int32)、通道名、相对时间(秒)、原始事件样本号。
    times 和数组长度直接来自 MNE，不假定采样率或固定采样点数。
    """
    scale = OUTPUT_UNIT_SCALES[config.STAGE4_OUTPUT_UNIT]
    dtype = np.dtype(config.STAGE4_OUTPUT_DTYPE)
    data = (epochs.get_data(copy=True) * scale).astype(dtype)
    labels = epochs.events[:, 2].astype(np.int32)
    names = np.asarray(epochs.ch_names, dtype="S")
    times = epochs.times.astype(np.float64)
    event_samples = epochs.events[:, 0].astype(np.int64)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite/overflowing {dtype} EEG export")
    return data, labels, names, times, event_samples


def attribute_value(value):
    """字典/序列元数据编码成 JSON 字符串；普通标量直接作为 HDF5 属性写入。"""
    return json.dumps(value) if isinstance(value, (dict, list, tuple)) else value


def write_hdf5(path, data, labels, names, times, event_samples, metadata):
    """仅写临时文件；正式发布由 publish_outputs 统一执行。"""
    with h5py.File(path, "w") as h5:
        for key, value in zip(("clean_eeg", "labels", "channel_names", "times", "event_samples"),
                              (data, labels, names, times, event_samples)):
            h5.create_dataset(key, data=value)
        for key, value in metadata.items():
            h5.attrs[key] = attribute_value(value)
    verify_hdf5(path, data, labels, names, times, event_samples, metadata)


def verify_hdf5(path, data, labels, names, times, event_samples, metadata):
    """回读全部数据和元数据，逐项对比，并调用现有 HDF5 消费端的格式校验。"""
    with h5py.File(path, "r") as h5:
        for key, value in zip(("clean_eeg", "labels", "channel_names", "times", "event_samples"),
                              (data, labels, names, times, event_samples)):
            if h5[key].dtype != value.dtype:
                raise RuntimeError(f"Stage 4 HDF5 dtype mismatch: {key}")
            np.testing.assert_array_equal(h5[key][:], value)
        if (h5["clean_eeg"].dtype != np.dtype(config.STAGE4_OUTPUT_DTYPE)
                or h5["labels"].dtype != np.dtype("int32")):
            raise RuntimeError("Unexpected stage 4 EEG or label dtype")
        for key, value in metadata.items():
            if h5.attrs[key] != attribute_value(value):
                raise RuntimeError(f"Stage 4 HDF5 attribute mismatch: {key}")
    validate_clean_hdf5(path)
    inspect_processed_file(Path(path))


def publish_outputs(full, ws, metadata, report, source):
    """逐文件原子替换，报告最后提交；可捕获异常回滚，不是跨文件系统事务。

    禁止并发运行/读取发布过程。断电或进程强杀无法回滚，须对照报告哈希
    判断完整性并重新运行。若回滚也失败，保留临时备份供人工恢复。
    """
    paths = output_paths()
    folders, temporary, backups, replaced = [], [], {}, []
    preserve = False
    try:
        # 第一步：同一目标磁盘上建立临时目录，并备份旧文件；此时正式输出未改变。
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix="stage4-", dir=path.parent))
            folders.append(folder)
            temporary.append(folder / path.name)
            if path.exists():
                if not config.STAGE4_OVERWRITE:
                    raise FileExistsError(path)
                backups[path] = folder / "previous.backup"
                shutil.copy2(path, backups[path])
        # 第二步：两个文件全部写完/回读通过，才生成引用它们哈希的报告。
        for path, bundle in zip(temporary[:2], (full, ws)):
            tmin, tmax = float(bundle[3][0]), float(bundle[3][-1])
            write_hdf5(path, *bundle, {**metadata, "export_window": f"{tmin:g}~{tmax:g}",
                                          "tmin": tmin, "tmax": tmax})
        report.update(full_sha256=file_hash(temporary[0]), ws_sha256=file_hash(temporary[1]))
        temporary[2].write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        if read_json(temporary[2]) != report:
            raise RuntimeError("Stage 4 report verification failed")
        # 第三步：再验上游依赖，然后逐个原子替换，报告作为最后的提交记录。
        verify_source(source)
        for staged, path in zip(temporary, paths):
            if path.exists() and not config.STAGE4_OVERWRITE:
                raise FileExistsError(path)
            os.replace(staged, path)
            replaced.append(path)
    except BaseException:
        # 逆序恢复已替换文件；首次发布没有旧文件时，删除本次新建输出即可。
        failures = []
        for path in reversed(replaced):
            try:
                if path in backups:
                    os.replace(backups[path], path)
                else:
                    path.unlink()
            except OSError as exc:
                failures.append(str(exc))
        if failures:
            preserve = True
            raise RuntimeError(f"Stage 4 rollback failed: {failures}; backups retained in {folders}")
        raise
    finally:
        if not preserve:
            for folder in folders:
                shutil.rmtree(folder)


def main():
    """运行完整阶段 4 并返回 (full_epochs, ws_epochs)；失败不会返回伪成功结果。"""
    validate_config()
    raw, source = load_clean_raw()
    filtered = filter_raw(raw)
    full_epochs = make_epochs(filtered)
    # 先用完整配置窗口审核坏段，再从通过审核的同一批试次裁出分类窗口。
    ws_epochs = full_epochs.copy().crop(
        tmin=config.STAGE4_WS_TMIN, tmax=config.STAGE4_WS_TMAX, include_tmax=True
    )
    rejection = rejection_report(full_epochs, raw)
    print_epoch_rejection_summary(full_epochs, ws_epochs, rejection)
    full = arrays(full_epochs)
    ws = arrays(ws_epochs)
    if full[0].shape[2] != len(full[3]) or ws[0].shape[2] != len(ws[3]):
        raise RuntimeError("Stage 4 data/time length mismatch")
    if not np.array_equal(full[1], ws[1]) or not np.array_equal(full[4], ws[4]):
        raise RuntimeError("Full and WS exports do not contain the same epochs")
    # 同批输出共享 run_id 和输入哈希；用于判断 full/ws 是否来自同一次处理。
    run_id = uuid.uuid4().hex
    metadata = {
        "stage": 4, "run_id": run_id, "sampling_rate": float(raw.info["sfreq"]),
        "unit": config.STAGE4_OUTPUT_UNIT, "axis_order": "epochs,channels,times",
        "filter_l_freq": config.STAGE4_FILTER_L_FREQ, "filter_h_freq": config.STAGE4_FILTER_H_FREQ,
        "epoch_tmin": config.STAGE4_EPOCH_TMIN, "epoch_tmax": config.STAGE4_EPOCH_TMAX,
        "source_raw_sha256": source["sha256"], "label_map": config.STAGE4_EVENT_ID,
        "source_raw_path": source["path"], "mne_version": mne.__version__,
        "filter_method": config.STAGE4_FILTER_METHOD, "filter_phase": config.STAGE4_FILTER_PHASE,
        "filter_design": config.STAGE4_FILTER_DESIGN,
        "filter_skip_annotations": list(config.STAGE4_FILTER_SKIP_ANNOTATIONS),
        "baseline": "None" if config.STAGE4_BASELINE is None else list(config.STAGE4_BASELINE),
        "reference": "average (already applied in stage 2)",
        "reject_by_annotation": True,
    }
    full_path, ws_path, _ = output_paths()
    counts = Counter(full[1].tolist())
    # 每个测试折都应包含每类样本；这里给出上限提示，不自动修改/运行分类器。
    max_folds = min(counts.values())
    notes = ["Assumes configured task annotations mark motor-imagery onset; confirm experimental timing."]
    if max_folds < 10:
        notes.append(f"ws default 10 folds is unsuitable: minimum class count={max_folds}. "
                     "Use 2 <= k_fold <= minimum class count; if below 2, collect more trials.")
    events, _ = mne.events_from_annotations(raw, event_id=config.STAGE4_EVENT_ID,
                                          regexp=None, verbose=False)
    min_gap = float(np.diff(events[:, 0]).min() / raw.info["sfreq"])
    if min_gap <= config.STAGE4_EPOCH_TMAX - config.STAGE4_EPOCH_TMIN:
        notes.append("Full epoch windows overlap; review event timing before classification.")
    report = {**metadata, "generated_at_utc": utc_now(), "full_output": str(full_path),
              "ws_output": str(ws_path), "full_shape": list(full[0].shape),
              "ws_shape": list(ws[0].shape), "rejection": rejection,
              "source": source, "amplitude_recheck": source["applied"]["amplitude_recheck"],
              "channel_names": raw.ch_names, "warnings": notes,
              "min_event_interval_seconds": min_gap, "max_stratified_k_fold": max_folds,
              "ws_tmin": float(ws[3][0]), "ws_tmax": float(ws[3][-1]),
              "annotations": dict(Counter(map(str, raw.annotations.description)))}
    publish_outputs(full, ws, metadata, report, source)
    for note in notes:
        warnings.warn(note, RuntimeWarning)
    print(f"Stage 4 complete: accepted={len(full_epochs)}, full={full[0].shape}, ws={ws[0].shape}")
    print(f"Published: {full_path}\nPublished: {ws_path}")
    return full_epochs, ws_epochs


if __name__ == "__main__":
    main()