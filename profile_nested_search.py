"""只读真实 EEG 的局部搜索剖析；不运行完整置换、不改写分类或置换结果。"""
import argparse
import cProfile
import ctypes
import io
import json
from pathlib import Path
import pstats
import tempfile
from time import perf_counter

import numpy as np
from joblib import Memory
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from mne.utils import use_log_level

import ws


def peak_working_set():
    """Windows 进程峰值工作集，含解释器/输入，不代表单独搜索的增量峰值。"""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in
            ("peak", "working", "quota_peak_paged", "quota_paged", "quota_peak_nonpaged",
             "quota_nonpaged", "pagefile", "peak_pagefile")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not get_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.peak / 1024**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("reference", "grouped"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.fold <= 10:
        raise ValueError("fold must be 1..10")
    data, labels = ws.load_evaluation_data()
    labels = labels[np.random.default_rng(42).permutation(len(labels))]
    train, test = list(StratifiedKFold(10).split(data, labels))[args.fold - 1]
    X, y = data[train], labels[train]
    grid = ws._search_grid(None, data.shape[1])
    profile = cProfile.Profile()
    with tempfile.TemporaryDirectory(prefix="bci-profile-") as cache, \
            threadpool_limits(limits=2), use_log_level("WARNING"):
        start = perf_counter()
        profile.enable()
        if args.engine == "reference":
            search = GridSearchCV(Pipeline([
                ("csp", ws._NativeFloat64CSP(n_components=2, reg="oas", log=False, norm_trace=False)),
                ("scaler", StandardScaler()), ("svm", SVC(kernel="linear", C=1.0)),
            ], memory=Memory(cache, verbose=0)), grid, cv=StratifiedKFold(5),
                scoring="accuracy", n_jobs=1, error_score="raise")
            search.fit(X, y)
            params = search.cv_results_["params"]
            scores = np.array([search.cv_results_[f"split{i}_test_score"] for i in range(5)]).T
            best = search.best_index_
            model = search.best_estimator_
        else:
            result = ws.grouped_search(X, y, grid, inner_k_fold=5, native_threads=2)
            params, scores, best = result["params"], result["split_scores"], result["best_index"]
            model = result["best_estimator"]
        predictions = model.predict(data[test])
        profile.disable()
        elapsed = perf_counter() - start
        cache_bytes = sum(p.stat().st_size for p in Path(cache).rglob("*") if p.is_file())
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(35)
    report = dict(engine=args.engine, outer_fold=args.fold, elapsed_seconds=elapsed,
                  peak_process_working_set_mib=peak_working_set(), cache_bytes=cache_bytes,
                  params=params, split_scores=scores.tolist(), best_index=int(best),
                  best_params=params[best], predictions=predictions.tolist(),
                  outer_score=float(np.mean(predictions == labels[test])), profile=stream.getvalue())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print({k: v for k, v in report.items() if k not in ("params", "split_scores", "profile")})
    print(stream.getvalue())


if __name__ == "__main__":
    main()