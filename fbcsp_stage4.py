"""阶段 4 FBCSP 输入生成器。

从阶段 3 发布的 ICA clean 连续 Raw 生成多个连续滤波频带，并使用同一
参考频带的 Epoch 质量控制结果保证所有频带的试次严格对齐。该模块只做
无标签滤波、事件分段和 HDF5 发布，不拟合 CSP 或训练分类器。
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

import h5py
import mne
import numpy as np

import config
import EEG_processing as stage4
from fbcsp import file_sha256, validate_fbcsp_hdf5


def _json_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_fbcsp_config(sfreq=None):
    stage4.validate_config(validate_outputs=False)
    values = np.asarray(config.FBCSP_BANDS)
    if values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in "fiu":
        raise ValueError("FBCSP_BANDS must be a numeric (bands,2) array")
    bands = tuple(tuple(float(value) for value in band) for band in values)
    if not bands:
        raise ValueError("FBCSP_BANDS must not be empty")
    previous_high = 0.0
    for low, high in bands:
        if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
            raise ValueError("Each FBCSP band must satisfy 0 < low < high")
        if low < previous_high:
            raise ValueError("FBCSP bands must be ordered and non-overlapping")
        previous_high = high
        if sfreq is not None and high >= float(sfreq) / 2:
            raise ValueError("FBCSP band high frequency must be below Nyquist")
        if low < config.STAGE4_FILTER_L_FREQ or high > config.STAGE4_FILTER_H_FREQ:
            raise ValueError("For matched baseline comparison, bands must lie within the reference band")
    paths = [Path(config.FBCSP_FULL_OUTPUT_PATH).resolve(),
             Path(config.FBCSP_WS_OUTPUT_PATH).resolve(),
             Path(config.FBCSP_BASELINE_OUTPUT_PATH).resolve(),
             Path(config.FBCSP_REPORT_PATH).resolve()]
    if len(set(paths)) != 4:
        raise ValueError("FBCSP outputs must be distinct")
    protected = {
        Path(config.STAGE4_INPUT_RAW_PATH).resolve(),
        Path(config.ICA_INPUT_RAW_PATH).resolve(),
        Path(config.ICA_MODEL_PATH).resolve(),
        Path(config.ICA_MANIFEST_PATH).resolve(),
        Path(config.ICA_CLEAN_RAW_PATH).resolve(),
        Path(config.STAGE4_FULL_OUTPUT_PATH).resolve(),
        Path(config.STAGE4_WS_OUTPUT_PATH).resolve(),
        Path(config.STAGE4_REPORT_PATH).resolve(),
    }
    if protected.intersection(paths):
        raise ValueError("FBCSP outputs must not overwrite upstream or CSP stage4 files")
    if type(config.FBCSP_OVERWRITE) is not bool:
        raise ValueError("FBCSP_OVERWRITE must be bool")
    if any(p.suffix != ".hdf5" for p in paths[:3]) or paths[3].suffix != ".json":
        raise ValueError("Expected three HDF5 files and one JSON report")
    if not config.FBCSP_OVERWRITE and any(p.exists() for p in paths):
        raise FileExistsError("FBCSP outputs exist; explicitly enable FBCSP_OVERWRITE")
    return bands, paths


def filter_band(raw, low, high):
    """在连续 Raw 副本上滤波，不修改 clean Raw。"""
    filtered = raw.copy()
    filtered.filter(
        l_freq=low,
        h_freq=high,
        picks="eeg",
        method=config.STAGE4_FILTER_METHOD,
        phase=config.STAGE4_FILTER_PHASE,
        fir_design=config.STAGE4_FILTER_DESIGN,
        skip_by_annotation=config.STAGE4_FILTER_SKIP_ANNOTATIONS,
        verbose=False,
    )
    if filtered.ch_names != raw.ch_names or filtered.n_times != raw.n_times:
        raise RuntimeError("FBCSP filtering changed Raw shape")
    for key in ("onset", "duration", "description"):
        np.testing.assert_array_equal(getattr(filtered.annotations, key), getattr(raw.annotations, key))
    if not np.isfinite(filtered.get_data()).all():
        raise RuntimeError("FBCSP filtering produced non-finite data")
    stage4.check_average_reference(filtered)
    return filtered


def _add_edge_aliases(raw):
    """复制阶段 4 的 edge→bad 处理到独立频带 Epoch 副本。"""
    result = raw.copy()
    for annotation in list(result.annotations):
        if annotation["description"].lower().startswith("edge"):
            result.annotations.append(
                annotation["onset"], annotation["duration"], "bad_fbcsp_edge"
            )
    return result


def _all_events(raw):
    events, _ = mne.events_from_annotations(
        raw, event_id=config.STAGE4_EVENT_ID, regexp=None, verbose=False
    )
    if len(events) == 0:
        raise RuntimeError("No configured task events available")
    return events


def _aligned_epochs(filtered, selected_events):
    """用参考分支确定的事件切 Epoch，不再次按频带独立删除试次。"""
    prepared = _add_edge_aliases(filtered)
    epochs = mne.Epochs(
        prepared,
        events=selected_events,
        event_id=config.STAGE4_EVENT_ID,
        tmin=config.STAGE4_EPOCH_TMIN,
        tmax=config.STAGE4_EPOCH_TMAX,
        baseline=config.STAGE4_BASELINE,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        event_repeated="error",
        verbose=False,
    )
    if len(epochs) != len(selected_events):
        raise RuntimeError("An aligned FBCSP band lost an event during Epoch creation")
    np.testing.assert_array_equal(epochs.events, selected_events)
    return epochs


def _stack_bands(band_epochs, tmin, tmax):
    arrays = []
    labels = None
    samples = None
    names = None
    times = None
    for epochs in band_epochs:
        cropped = epochs.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True)
        data, current_labels, current_names, current_times, current_samples = stage4.arrays(cropped)
        if labels is None:
            labels, samples, names, times = current_labels, current_samples, current_names, current_times
        else:
            np.testing.assert_array_equal(current_labels, labels)
            np.testing.assert_array_equal(current_samples, samples)
            np.testing.assert_array_equal(current_names, names)
            np.testing.assert_array_equal(current_times, times)
        arrays.append(data)
    data = np.stack(arrays, axis=1)
    return data, labels, names, times, samples


def _write_hdf5(path, bundle, metadata, window):
    data, labels, names, times, samples, bands = bundle
    attributes = {**metadata, "export_window": window,
                  "tmin": float(times[0]), "tmax": float(times[-1])}
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("clean_eeg", data=data)
        h5_file.create_dataset("labels", data=labels)
        h5_file.create_dataset("channel_names", data=names)
        h5_file.create_dataset("times", data=times)
        h5_file.create_dataset("event_samples", data=samples)
        h5_file.create_dataset("bands", data=np.asarray(bands, dtype=np.float64))
        for key, value in attributes.items():
            h5_file.attrs[key] = _json_value(value)
    validate_fbcsp_hdf5(path)
    with h5py.File(path, "r") as check:
        for name, expected in zip(
                ("clean_eeg", "labels", "channel_names", "times", "event_samples", "bands"),
                bundle):
            np.testing.assert_array_equal(check[name][:], expected)
            if check[name].dtype != np.asarray(expected).dtype:
                raise RuntimeError(f"FBCSP HDF5 dtype mismatch: {name}")
        for key, value in attributes.items():
            if check.attrs[key] != _json_value(value):
                raise RuntimeError(f"FBCSP HDF5 attribute mismatch: {key}")


def _publish(full_bundle, ws_bundle, baseline_bundle, metadata, report, paths, source):
    folders = []
    temporary = []
    backups = {}
    replaced = []
    preserve = False
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            folder = Path(tempfile.mkdtemp(prefix="fbcsp-stage4-", dir=path.parent))
            folders.append(folder)
            temporary.append(folder / path.name)
            if path.exists():
                if not config.FBCSP_OVERWRITE:
                    raise FileExistsError(path)
                backups[path] = folder / "previous.backup"
                shutil.copy2(path, backups[path])
        _write_hdf5(temporary[0], full_bundle, metadata,
                    f"{full_bundle[3][0]:g}~{full_bundle[3][-1]:g}")
        _write_hdf5(temporary[1], ws_bundle, metadata,
                    f"{ws_bundle[3][0]:g}~{ws_bundle[3][-1]:g}")
        baseline_metadata = {**metadata, "axis_order": "epochs,channels,times",
                             "export_kind": "baseline_ws",
                             "filter_l_freq": config.STAGE4_FILTER_L_FREQ,
                             "filter_h_freq": config.STAGE4_FILTER_H_FREQ,
                             "tmin": float(baseline_bundle[3][0]),
                             "tmax": float(baseline_bundle[3][-1])}
        stage4.write_hdf5(temporary[2], *baseline_bundle, baseline_metadata)
        report = dict(report,
                      full_sha256=file_sha256(temporary[0]),
                      ws_sha256=file_sha256(temporary[1]),
                      baseline_sha256=file_sha256(temporary[2]))
        temporary[3].write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        if json.loads(temporary[3].read_text(encoding="utf-8")) != report:
            raise RuntimeError("FBCSP report verification failed")
        stage4.verify_source(source)
        for staged, path in zip(temporary, paths):
            if path.exists() and not config.FBCSP_OVERWRITE:
                raise FileExistsError(path)
            os.replace(staged, path)
            replaced.append(path)
        return report
    except BaseException:
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
            raise RuntimeError(f"FBCSP publish rollback failed: {failures}; backups={folders}")
        raise
    finally:
        if not preserve:
            for folder in folders:
                shutil.rmtree(folder, ignore_errors=True)


def main():
    validate_fbcsp_config()
    raw, source = stage4.load_clean_raw()
    bands, paths = validate_fbcsp_config(raw.info["sfreq"])
    if set(paths).intersection(Path(p).resolve() for p in source["dependencies"]):
        raise ValueError("FBCSP output would overwrite a source dependency")
    # 参考频带只用于统一质量控制和试次选择；其数据不作为 FBCSP 子带之一。
    reference = stage4.filter_raw(raw)
    reference_epochs = stage4.make_epochs(reference)
    all_reference_events = _all_events(reference)
    selected_events = all_reference_events[reference_epochs.selection]
    rejection = stage4.rejection_report(reference_epochs, raw)
    band_epochs = []
    for low, high in bands:
        filtered = filter_band(raw, low, high)
        band_events = _all_events(filtered)
        np.testing.assert_array_equal(band_events, all_reference_events)
        band_epochs.append(_aligned_epochs(filtered, selected_events))

    full_bundle_data = _stack_bands(
        band_epochs, config.STAGE4_EPOCH_TMIN, config.STAGE4_EPOCH_TMAX
    )
    ws_bundle_data = _stack_bands(
        band_epochs, config.STAGE4_WS_TMIN, config.STAGE4_WS_TMAX
    )
    full_bundle = (*full_bundle_data, bands)
    ws_bundle = (*ws_bundle_data, bands)
    baseline_bundle = stage4.arrays(reference_epochs.copy().crop(
        config.STAGE4_WS_TMIN, config.STAGE4_WS_TMAX, include_tmax=True))
    for index in (1, 2, 3, 4):
        np.testing.assert_array_equal(ws_bundle[index], baseline_bundle[index])
    if not np.array_equal(full_bundle[1], ws_bundle[1]) or not np.array_equal(
            full_bundle[4], ws_bundle[4]):
        raise RuntimeError("FBCSP full and WS outputs do not contain identical epochs")

    run_id = uuid.uuid4().hex
    metadata = {
        "stage": 4,
        "run_id": run_id,
        "source_raw_sha256": source["sha256"],
        "source_raw_path": source["path"],
        "sampling_rate": float(raw.info["sfreq"]),
        "unit": config.STAGE4_OUTPUT_UNIT,
        "axis_order": "epochs,bands,channels,times",
        "label_map": config.STAGE4_EVENT_ID,
        "epoch_tmin": config.STAGE4_EPOCH_TMIN,
        "epoch_tmax": config.STAGE4_EPOCH_TMAX,
        "classification_tmin": config.STAGE4_WS_TMIN,
        "classification_tmax": config.STAGE4_WS_TMAX,
        "output_dtype": str(np.dtype(config.STAGE4_OUTPUT_DTYPE)),
        "schema_version": 1,
        "bands": [list(band) for band in bands],
        "mne_version": mne.__version__,
        "filter_method": config.STAGE4_FILTER_METHOD,
        "filter_phase": config.STAGE4_FILTER_PHASE,
        "filter_design": config.STAGE4_FILTER_DESIGN,
        "filter_skip_annotations": list(config.STAGE4_FILTER_SKIP_ANNOTATIONS),
        "baseline": "None" if config.STAGE4_BASELINE is None else list(config.STAGE4_BASELINE),
        "reference": "average (already applied in stage 2)",
        "reject_by_annotation": True,
        "reference_filter_band": [config.STAGE4_FILTER_L_FREQ, config.STAGE4_FILTER_H_FREQ],
    }
    counts = Counter(full_bundle[1].tolist())
    report = {
        **metadata,
        "generated_at_utc": utc_now(),
        "full_output": str(paths[0]), "ws_output": str(paths[1]),
        "baseline_output": str(paths[2]),
        "full_shape": list(full_bundle[0].shape), "ws_shape": list(ws_bundle[0].shape),
        "rejection": rejection,
        "source": source,
        "channel_names": raw.ch_names,
        "class_counts": {str(key): int(value) for key, value in counts.items()},
        "reference_filter_band": [config.STAGE4_FILTER_L_FREQ, config.STAGE4_FILTER_H_FREQ],
        "alignment": "all bands use reference branch accepted event_samples",
    }
    report = _publish(full_bundle, ws_bundle, baseline_bundle, metadata, report, paths, source)
    print(f"FBCSP stage 4 complete: full={full_bundle[0].shape}, ws={ws_bundle[0].shape}")
    print(f"Published: {paths[0]}\nPublished: {paths[1]}\nBaseline: {paths[2]}\nReport: {paths[3]}")
    return full_bundle, ws_bundle, report


if __name__ == "__main__":
    main()