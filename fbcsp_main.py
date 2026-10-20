"""FBCSP 与同源单频带 CSP 配对评估；新目录发布，绝不覆盖历史结果。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import tempfile

import h5py
import numpy as np
from scipy.io import savemat
import scipy

import config
import fbcsp
import ws


def versions():
    return {**ws.runtime_versions(), "python": platform.python_version(),
            "scipy": scipy.__version__, "h5py": h5py.__version__}


def save_json(path, record):
    """同目录临时文件写完再替换，供结果与置换检查点共用。"""
    path = Path(path)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=path.parent,
                                         encoding="utf-8", delete=False) as stream:
            name = stream.name
            json.dump(record, stream, ensure_ascii=True, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def load_publication(report_path=None):
    """只接受完整的 FBCSP 阶段4报告，同时核验配对 baseline。"""
    path = Path(report_path or config.FBCSP_REPORT_PATH).resolve()
    digest = fbcsp.file_sha256(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("axis_order") != "epochs,bands,channels,times" or report.get("stage") != 4:
        raise ValueError("Not an FBCSP stage4 report")
    dependencies = {str(path): digest}
    for kind in ("full", "ws", "baseline"):
        output = Path(report[f"{kind}_output"]).resolve()
        expected = report[f"{kind}_sha256"]
        if fbcsp.file_sha256(output) != expected:
            raise ValueError(f"Stage4 {kind} SHA256 mismatch")
        dependencies[str(output)] = expected
        if kind == "baseline":
            ws.validate_clean_hdf5(output)
        else:
            fbcsp.validate_fbcsp_hdf5(output)
        with h5py.File(output, "r") as h5:
            for key in ("run_id", "source_raw_sha256", "sampling_rate", "unit"):
                if h5.attrs[key] != report[key]:
                    raise ValueError(f"Stage4 {kind} metadata mismatch: {key}")
            if kind != "baseline":
                np.testing.assert_array_equal(h5["bands"][:], report["bands"])
    with h5py.File(report["ws_output"], "r") as fb, \
            h5py.File(report["baseline_output"], "r") as base, \
            h5py.File(report["full_output"], "r") as full:
        for key in ("labels", "event_samples", "channel_names"):
            np.testing.assert_array_equal(fb[key][:], base[key][:])
            np.testing.assert_array_equal(fb[key][:], full[key][:])
        np.testing.assert_array_equal(fb["times"][:], base["times"][:])
        n_epochs, _, n_channels, n_times = fb["clean_eeg"].shape
        if (n_epochs, n_channels, n_times) != base["clean_eeg"].shape:
            raise ValueError("Baseline EEG dimensions do not match filter bank")
        if fb["clean_eeg"].shape[:3] != full["clean_eeg"].shape[:3]:
            raise ValueError("Full/WS filter bank dimensions differ")
        start = int(np.argmin(np.abs(full["times"][:] - fb["times"][0])))
        stop = start + len(fb["times"])
        np.testing.assert_array_equal(full["times"][start:stop], fb["times"][:])
        for i in range(0, len(fb["labels"]), 8):
            np.testing.assert_array_equal(full["clean_eeg"][i:i + 8, :, :, start:stop],
                                          fb["clean_eeg"][i:i + 8])
        np.testing.assert_array_equal([base.attrs["filter_l_freq"], base.attrs["filter_h_freq"]],
                                      report["reference_filter_band"])
    verify_dependencies(dependencies)
    return report, dependencies


def verify_dependencies(dependencies):
    for path, digest in dependencies.items():
        if fbcsp.file_sha256(path) != digest:
            raise RuntimeError(f"Input changed: {path}")


def run_classification(*, report_path=None, output_dir=None, k_fold=10, inner_k_fold=5,
                       param_grid=None, n_jobs=1, native_threads=2, include_baseline=True):
    if type(include_baseline) is not bool:
        raise TypeError("include_baseline must be bool")
    report, dependencies = load_publication(report_path)
    data, labels, metadata = fbcsp.load_fbcsp_data(report["ws_output"])
    verify_dependencies(dependencies)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path(output_dir or config.PROJECT_ROOT / "FBCSP_result" / stamp).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    options = dict(k_fold=k_fold, inner_k_fold=inner_k_fold, param_grid=param_grid,
                   n_jobs=n_jobs, native_threads=native_threads, return_details=True)
    accuracy, scores, details = fbcsp.nested_cv(data, labels, **options)
    result = {"model": "FBCSP_SVM", "status": "completed", "accuracy": accuracy,
              "fold_scores": scores, "cv_details": details, "metrics": fbcsp.metrics(details),
              "input": metadata, "dependencies": dependencies, "runtime_versions": versions(),
              "stage4_report": str(Path(report_path or config.FBCSP_REPORT_PATH).resolve()),
              "feature_selection": "none"}
    if include_baseline:
        base, base_labels = ws.load_clean_data(report["baseline_output"])
        np.testing.assert_array_equal(base_labels, labels)
        a, s, d = fbcsp.nested_cv(base[:, None], labels, baseline=True, **options)
        for fold_a, fold_b in zip(details["folds"], d["folds"]):
            if fold_a["test_indices"] != fold_b["test_indices"]:
                raise RuntimeError("Paired folds differ")
        result["baseline"] = {"accuracy": a, "fold_scores": s, "cv_details": d,
                              "metrics": fbcsp.metrics(d)}
        result["accuracy_difference"] = accuracy - a
    verify_dependencies(dependencies)
    # JSON 是完整结果/完成标记，最后发布；置换入口只消费此文件。
    savemat(directory / "fbcsp_classification_result.mat", {
        "accuracy": accuracy, "fold_scores": scores,
        "result_json": json.dumps(result, allow_nan=False)})
    (directory / "results.txt").write_text(
        f"FBCSP accuracy: {accuracy:.12g}\nBands: {metadata['bands']}\n"
        f"Fold scores: {scores}\nMetrics: {result['metrics']}\n"
        f"Paired difference: {result.get('accuracy_difference', 'not evaluated')}\n",
        encoding="utf-8")
    save_json(directory / "classification_result.json", result)
    print(f"Published FBCSP classification: {directory}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--native-threads", type=int, default=2)
    parser.add_argument("--fbcsp-only", action="store_true")
    args = parser.parse_args(argv)
    return run_classification(report_path=args.stage4_report, output_dir=args.output_dir,
                              n_jobs=args.n_jobs, native_threads=args.native_threads,
                              include_baseline=not args.fbcsp_only)


if __name__ == "__main__":
    main()