"""Epoch 级标签置换检验；复用已保存的原始分数，每次完整重跑嵌套选参。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import warnings
import os
from time import perf_counter

import h5py
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.model_selection import ParameterGrid, StratifiedKFold

import config
from main import DEFAULT_RESULT_MAT_PATH
import ws


DEFAULT_N_PERMUTATIONS = 100
DEFAULT_RANDOM_SEED = 42
DEFAULT_OUTPUT_ROOT = config.PROJECT_ROOT / "CSP_result" / "permutation"


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def empirical_p_value(observed_score, permutation_scores):
    """准确率单侧 Monte Carlo P 值，包含并列值，使用 +1 修正。"""
    values = np.asarray(permutation_scores, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
            or not np.isfinite(observed_score) or not 0 <= observed_score <= 1):
        raise ValueError("Scores must be finite accuracies in [0, 1], with at least one permutation")
    # 相同有理数准确率的浮点求和舍入差异按并列处理。
    extreme = (values >= observed_score) | np.isclose(values, observed_score, rtol=0, atol=1e-12)
    count = int(np.count_nonzero(extreme))
    return (count + 1) / (len(values) + 1), count


def load_observed_result(observed_mat_path=None, hdf5_path=None, stage4_report_path=None):
    """只读校验原始 MAT 及输入身份，不重新训练，也不使用 TXT 的四位小数分数。

    新结果以输入 SHA256 为凭据。旧结果缺少运行时输入哈希时，仅接受具有匹配
    run_id/输出路径/文件哈希的阶段4报告；这不能追溯证明历史运行环境完全相同。
    """
    mat_path = Path(observed_mat_path or DEFAULT_RESULT_MAT_PATH).expanduser().resolve()
    mat_hash = ws.file_sha256(mat_path)
    stored = loadmat(mat_path, simplify_cells=True)
    settings = stored["config"]
    details = json.loads(stored["nested_cv_details_json"])
    path = Path(settings["clean_data_path"] if hdf5_path is None else hdf5_path).expanduser().resolve()
    if path != Path(settings["clean_data_path"]).expanduser().resolve():
        raise ValueError("Observed result input path mismatch")
    input_hash = ws.file_sha256(path)
    ws.validate_clean_hdf5(path)
    with h5py.File(path, "r") as f:
        labels = f["labels"][:]
        shape = tuple(f["clean_eeg"].shape)
        attrs = {key: ws._text(f.attrs[key]) for key in ("run_id", "unit", "label_map")}
        if f.attrs.get("stage") != 4 or settings.get("stage") != 4:
            raise ValueError("Permutation entry requires formal stage 4 input/results")
        for key in ("sampling_rate", "tmin", "tmax"):
            if float(settings[key]) != float(f.attrs[key]):
                raise ValueError(f"Observed result metadata mismatch: {key}")
    values, counts = np.unique(labels, return_counts=True)
    if (attrs["run_id"] != settings["run_id"] or attrs["unit"] != settings["unit"]
            or tuple(np.atleast_1d(settings["data_shape"])) != shape
            or json.loads(settings["label_map_json"]) != json.loads(attrs["label_map"])
            or json.loads(settings["class_counts_json"]) != dict(zip(map(str, values), map(int, counts)))):
        raise ValueError("Observed result input identity mismatch")

    provenance = "classification_input_sha256"
    if "input_sha256" in settings:
        if settings["input_sha256"] != input_hash:
            raise ValueError("Observed result input SHA256 mismatch")
    else:
        report_path = Path(stage4_report_path or config.STAGE4_REPORT_PATH).expanduser().resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["run_id"] != attrs["run_id"] or report["ws_sha256"] != input_hash
                or Path(report["ws_output"]).resolve() != path):
            raise ValueError("Legacy observed result stage4 report/hash mismatch")
        provenance = "legacy_stage4_report"
        warnings.warn("Legacy classification lacks input/runtime fingerprints; verified stage4 "
                      "report, run_id and current input hash. Historical runtime cannot be proven.",
                      RuntimeWarning)

    if (settings.get("cv_protocol", ws.CV_PROTOCOL) != ws.CV_PROTOCOL
            or details.get("cv_protocol", ws.CV_PROTOCOL) != ws.CV_PROTOCOL):
        raise ValueError("Observed CV protocol mismatch")
    if "runtime_versions_json" in settings:
        if json.loads(settings["runtime_versions_json"]) != ws.runtime_versions():
            raise ValueError("Observed runtime versions mismatch; do not mix experimental environments")
    for record in (settings, details):
        if (record["cv_method"] != "nested_stratified_k_fold" or record["scoring"] != "accuracy"
                or record["shuffle"] != False):
            raise ValueError("Observed result CV method/scoring/shuffle mismatch")
    outer, inner = details["outer_k_fold"], details["inner_k_fold"]
    ws._validate_fold_count(outer, "outer_k_fold")
    ws._validate_fold_count(inner, "inner_k_fold")
    if settings["k_fold"] != outer or settings["inner_k_fold"] != inner:
        raise ValueError("Observed result fold count mismatch")
    grid = ws._search_grid(details["param_grid"], shape[1])
    if (grid != json.loads(settings["param_grid_json"])
            or details["n_candidates"] != len(ParameterGrid(grid))
            or details["pipeline_defaults"] != ws.PIPELINE_DEFAULTS):
        raise ValueError("Observed result grid/pipeline mismatch")
    # 采用保存的实际网格，不受以后 DEFAULT_PARAM_GRID 的修改影响。
    fold_scores = np.atleast_1d(stored["fold_scores"]).astype(float)
    score = float(stored["accuracy"])
    if (fold_scores.shape != (outer,) or not np.isfinite(fold_scores).all()
            or np.any((fold_scores < 0) | (fold_scores > 1))
            or not np.isfinite(score) or not np.isclose(score, fold_scores.mean(), rtol=0, atol=1e-12)
            or len(details["folds"]) != outer):
        raise ValueError("Observed accuracy/fold scores mismatch")
    if min(counts) < outer:
        raise ValueError("Observed outer fold count exceeds class count")
    splits = StratifiedKFold(outer, shuffle=False).split(np.zeros(len(labels)), labels)
    for index, ((train, test), fold) in enumerate(zip(splits, details["folds"])):
        if (fold["fold"] != index + 1 or fold["train_indices"] != train.tolist()
                or fold["test_indices"] != test.tolist()
                or not np.isclose(fold["outer_score"], fold_scores[index], rtol=0, atol=1e-12)):
            raise ValueError("Observed fold indices/scores mismatch")
        if min(np.unique(labels[train], return_counts=True)[1]) < inner:
            raise ValueError("Observed inner folds exceed training class count")
    band = np.atleast_1d(settings["filter_band"]).astype(float).tolist()
    if bool(settings["need_filter"]) != bool(band):
        raise ValueError("Observed filter configuration mismatch")
    if ws.file_sha256(path) != input_hash or ws.file_sha256(mat_path) != mat_hash:
        raise RuntimeError("Input or observed result changed during validation")
    return {
        "observed_mat_path": str(mat_path), "observed_mat_sha256": mat_hash,
        "input_path": str(path), "input_sha256": input_hash, "run_id": attrs["run_id"],
        "provenance": provenance, "observed_score": score,
        "observed_fold_scores": fold_scores.tolist(), "observed_cv_details": details,
        "filter_band": band or None, "runtime_versions": ws.runtime_versions(),
    }


def _save_checkpoint(path, record):
    """原子替换，避免中断写坏之前已完成的置换记录。"""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="checkpoint-", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(record, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_resume_record(record, observed, labels, n_permutations, random_seed):
    """只读核验并重放 RNG，返回恰好位于下一次置换前的生成器。"""
    for key in ("input_sha256", "observed_mat_sha256", "input_path", "observed_mat_path",
                "observed_score", "observed_fold_scores", "observed_cv_details",
                "filter_band", "runtime_versions", "run_id"):
        if record.get(key) != observed[key]:
            raise ValueError(f"Resume identity mismatch: {key}")
    required = {"n_permutations": n_permutations, "random_seed": random_seed,
                "permutation_unit": "epoch", "permutation_scheme": "global_labels_only",
                "split_policy": "restratify_for_each_permuted_label_vector",
                "rng": "numpy.default_rng/PCG64"}
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(f"Resume configuration mismatch: {key}")
    if record.get("status") not in ("failed", "running", "completed"):
        raise ValueError("Invalid resume status")
    completed = _integer(record["n_completed"], "n_completed", 0)
    if completed > n_permutations or len(record["permutations"]) != completed:
        raise ValueError("Resume completed count mismatch")
    if record["status"] == "completed" and completed != n_permutations:
        raise ValueError("Completed checkpoint has missing permutations")
    rng = np.random.default_rng(random_seed)
    original = observed["observed_cv_details"]
    for index, item in enumerate(record["permutations"], start=1):
        indices = rng.permutation(len(labels))
        permuted = labels[indices]
        if (item["iteration"] != index or item["permutation_indices"] != indices.tolist()
                or item["permuted_labels"] != permuted.tolist()):
            raise ValueError(f"Resume permutation sequence mismatch at {index}")
        cv = item["cv_details"]
        for key in ("cv_method", "outer_k_fold", "inner_k_fold", "shuffle", "scoring",
                    "param_grid", "n_candidates", "pipeline_defaults"):
            if cv[key] != original[key]:
                raise ValueError(f"Resume CV mismatch: {key}")
        scores = np.asarray(item["fold_scores"], dtype=float)
        outer = original["outer_k_fold"]
        if (scores.shape != (outer,) or not np.isfinite(scores).all()
                or np.any((scores < 0) | (scores > 1)) or len(cv["folds"]) != outer
                or not np.isclose(item["accuracy"], scores.mean(), rtol=0, atol=1e-12)):
            raise ValueError("Resume scores mismatch")
        splits = StratifiedKFold(outer, shuffle=False).split(np.zeros(len(labels)), permuted)
        candidates = list(ParameterGrid(original["param_grid"]))
        for fold_index, ((train, test), fold) in enumerate(zip(splits, cv["folds"])):
            if (fold["fold"] != fold_index + 1 or fold["train_indices"] != train.tolist()
                    or fold["test_indices"] != test.tolist()
                    or fold["outer_score"] != float(scores[fold_index])
                    or fold["best_params"] not in candidates
                    or not 0 <= fold["best_inner_score"] <= 1):
                raise ValueError("Resume fold details mismatch")
    if record["status"] == "completed":
        p_value, count = empirical_p_value(observed["observed_score"],
                                          [item["accuracy"] for item in record["permutations"]])
        if record.get("p_value") != p_value or record.get("n_extreme") != count:
            raise ValueError("Resume final P value mismatch")
    return rng


def run_permutation_test(**kwargs):
    """运行新实验或续跑；续跑锁防止两个新版进程同时改写同一检查点。"""
    resume = kwargs.get("resume_from")
    if resume is None:
        return _run_permutation_test(**kwargs)
    checkpoint = Path(resume).expanduser().resolve()
    lock_path = checkpoint.with_suffix(".lock")
    # 崩溃遗留锁须确认没有活跃任务后手工删除，不能自动抢占。
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    try:
        return _run_permutation_test(**kwargs)
    finally:
        lock_path.unlink(missing_ok=True)


def _run_permutation_test(*, n_permutations=DEFAULT_N_PERMUTATIONS,
                         random_seed=DEFAULT_RANDOM_SEED, observed_mat_path=None,
                         hdf5_path=None, stage4_report_path=None, output_dir=None, n_jobs=1,
                         resume_from=None, cache_transformers=True, native_threads=2,
                         search_engine="grouped"):
    """每次原子保存；可显式续跑，不根据中途 P 值停止或改变 N。"""
    n_permutations = _integer(n_permutations, "n_permutations", 1)
    random_seed = _integer(random_seed, "random_seed", 0)
    ws._validate_cv_options(10, 5, n_jobs, False)
    if search_engine not in ("reference", "grouped"):
        raise ValueError("search_engine must be reference or grouped")
    native_threads = _integer(native_threads, "native_threads", 1)
    if type(cache_transformers) is not bool:
        raise TypeError("cache_transformers must be True or False")
    if resume_from is not None and output_dir is not None:
        raise ValueError("resume_from cannot be combined with output_dir")
    observed = load_observed_result(observed_mat_path, hdf5_path, stage4_report_path)
    data, labels = ws.load_evaluation_data(observed["input_path"], observed["filter_band"])
    if ws.file_sha256(observed["input_path"]) != observed["input_sha256"]:
        raise RuntimeError("Input changed during loading")
    # Pipeline 只获得按折索引复制的数据，源 EEG 和原标签均设为只读。
    data.setflags(write=False)
    labels.setflags(write=False)
    details = observed["observed_cv_details"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if resume_from is None:
        directory = Path(output_dir if output_dir is not None else DEFAULT_OUTPUT_ROOT / timestamp).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = directory / "permutation_results.json"
    else:
        checkpoint = Path(resume_from).expanduser().resolve()
        directory = checkpoint.parent
        if checkpoint.name != "permutation_results.json":
            raise ValueError("Resume requires a permutation_results.json checkpoint")
    record = {
        **observed, "created_at_utc": timestamp, "status": "running",
        "permutation_unit": "epoch", "permutation_scheme": "global_labels_only",
        "split_policy": "restratify_for_each_permuted_label_vector",
        "n_permutations": n_permutations, "n_completed": 0, "random_seed": random_seed,
        "rng": "numpy.default_rng/PCG64", "n_jobs": int(n_jobs),
        "p_value": None, "n_extreme": None,
        "p_value_formula": "(1 + count(permuted_accuracy >= observed_accuracy)) / (N + 1)",
        "tie_atol": 1e-12, "permutations": [],
    }
    rng = np.random.default_rng(random_seed)
    if resume_from is not None:
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        rng = validate_resume_record(record, observed, labels, n_permutations, random_seed)
        if record["status"] == "completed":
            print("Checkpoint already completed; no fitting or rewriting required")
            return record
        print(f"Resuming after {record['n_completed']} completed permutations", flush=True)
    start_completed = record["n_completed"]
    record.setdefault("execution_sessions", []).append({
        "started_at_utc": timestamp, "start_completed": start_completed,
        "n_jobs": int(n_jobs), "native_threads": native_threads,
        "cache_transformers": cache_transformers if search_engine == "reference" else False,
        "search_engine": search_engine,
    })
    record.update(status="running", p_value=None, n_extreme=None, n_jobs=int(n_jobs))
    record.pop("error", None)
    record.setdefault("permutation_timings", [])
    _save_checkpoint(checkpoint, record)
    print(f"Observed accuracy: {observed['observed_score']:.6f}; epoch permutations: {n_permutations}")
    print(f"Checkpoint: {checkpoint}; search engine: {search_engine}")
    session_started = perf_counter()
    try:
        for iteration in range(start_completed + 1, n_permutations + 1):
            # 每次独立从原始标签取样；不对 EEG 使用 permutation_indices。
            permutation_indices = rng.permutation(len(labels))
            permuted_labels = labels[permutation_indices]
            print(f"Permutation {iteration}/{n_permutations}", flush=True)
            iteration_started = perf_counter()
            score, scores, cv_details = ws.nested_cv(
                data, permuted_labels, k_fold=details["outer_k_fold"],
                inner_k_fold=details["inner_k_fold"], param_grid=details["param_grid"],
                n_jobs=n_jobs, return_details=True,
                cache_transformers=cache_transformers, native_threads=native_threads,
                search_engine=search_engine,
            )
            record["permutations"].append({
                "iteration": iteration, "permutation_indices": permutation_indices.tolist(),
                "permuted_labels": permuted_labels.tolist(), "accuracy": score,
                "fold_scores": scores, "cv_details": cv_details,
            })
            record["n_completed"] = iteration
            elapsed = perf_counter() - iteration_started
            record["permutation_timings"].append({"iteration": iteration, "elapsed_seconds": elapsed})
            _save_checkpoint(checkpoint, record)
            mean_seconds = (perf_counter() - session_started) / (iteration - start_completed)
            eta_minutes = mean_seconds * (n_permutations - iteration) / 60
            print(f"Permutation {iteration}: accuracy={score:.6f}; saved {iteration}/{n_permutations}; "
                  f"elapsed={elapsed:.1f}s; estimated remaining={eta_minutes:.1f} min", flush=True)
        # 防止长时间运行中输入/观测结果被其他进程替换后误报最终 P 值。
        if (ws.file_sha256(observed["input_path"]) != observed["input_sha256"]
                or ws.file_sha256(observed["observed_mat_path"]) != observed["observed_mat_sha256"]):
            raise RuntimeError("Input or observed result changed during permutation testing")
        permutation_scores = [item["accuracy"] for item in record["permutations"]]
        p_value, n_extreme = empirical_p_value(observed["observed_score"], permutation_scores)
        savemat(directory / "permutation_summary.mat", {
            "observed_accuracy": observed["observed_score"],
            "observed_fold_scores": observed["observed_fold_scores"],
            "permutation_scores": permutation_scores,
            "permutation_fold_scores": [item["fold_scores"] for item in record["permutations"]],
            "p_value": p_value, "n_extreme": n_extreme, "n_permutations": n_permutations,
            "random_seed": random_seed, "permutation_unit": "epoch",
        })
        with (directory / "permutation_summary.txt").open("w", encoding="utf-8") as f:
            f.write(f"Observed accuracy: {observed['observed_score']:.12g}\n")
            f.write(f"Permutation unit: epoch (labels only)\nRandom seed: {random_seed}\n")
            for item in record["permutations"]:
                f.write(f"Permutation {item['iteration']}: {item['accuracy']:.12g}; folds={item['fold_scores']}\n")
            f.write(f"N: {n_permutations}\nExtreme count (>=, ties included): {n_extreme}\n")
            f.write(f"Empirical P value: ({n_extreme}+1)/({n_permutations}+1) = {p_value:.12g}\n")
        record.update(status="completed", p_value=p_value, n_extreme=n_extreme)
        _save_checkpoint(checkpoint, record)
    except BaseException as error:
        record.update(status="failed", p_value=None, n_extreme=None,
                      error=f"{type(error).__name__}: {error}")
        _save_checkpoint(checkpoint, record)
        raise
    print(f"Empirical P value: {record['p_value']:.6g}; completed {n_permutations} permutations")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--observed-mat", type=Path, default=DEFAULT_RESULT_MAT_PATH)
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--stage4-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--native-threads", type=int, default=2, help="每个进程的底层线程上限")
    parser.add_argument("--search-engine", choices=("grouped", "reference"), default="grouped",
                        help="默认grouped折内分组搜索；reference为原GridSearchCV")
    parser.add_argument("--no-cache", action="store_true", help="仅reference：关闭 Pipeline 磁盘缓存")
    parser.add_argument("--resume", type=Path, help="校验并续跑 permutation_results.json，N/seed必须一致")
    parser.add_argument("--validate-only", action="store_true", help="只读核验原始结果，不拟合、不写文件")
    args = parser.parse_args(argv)
    if args.validate_only:
        result = load_observed_result(args.observed_mat, args.hdf5, args.stage4_report)
        if args.resume:
            with h5py.File(result["input_path"], "r") as f:
                labels = f["labels"][:]
            record = json.loads(args.resume.read_text(encoding="utf-8"))
            validate_resume_record(record, result, labels, args.n_permutations, args.seed)
            print(f"Resume checkpoint valid: {record['n_completed']}/{record['n_permutations']}")
        print(f"Observed accuracy: {result['observed_score']:.12g}")
        print(f"Input SHA256: {result['input_sha256']}")
        print(f"Provenance: {result['provenance']}; validation only, no permutations executed")
        return result
    return run_permutation_test(n_permutations=args.n_permutations, random_seed=args.seed,
                                observed_mat_path=args.observed_mat, hdf5_path=args.hdf5,
                                stage4_report_path=args.stage4_report, output_dir=args.output_dir,
                                n_jobs=args.n_jobs, native_threads=args.native_threads,
                                cache_transformers=not args.no_cache, resume_from=args.resume,
                                search_engine=args.search_engine)


if __name__ == "__main__":
    main()