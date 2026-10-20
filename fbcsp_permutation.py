"""FBCSP-SVM 专用 Epoch 标签置换；每次完整重跑同步网格的嵌套 CV。

固定预处理不重跑。全局 Epoch 置换假定试次标签可交换；存在 run/session
结构时不能未经论证使用本检验。本检验不等同于 CSP/FBCSP 优劣检验。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid

import config
import fbcsp
from fbcsp_main import load_publication, save_json, verify_dependencies, versions


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def empirical_p_value(observed, scores):
    values = np.asarray(scores, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1)) or not np.isfinite(observed)
            or not 0 <= observed <= 1):
        raise ValueError("Invalid accuracy values")
    count = int(np.count_nonzero((values >= observed) | np.isclose(values, observed, rtol=0, atol=1e-12)))
    return (count + 1) / (len(values) + 1), count


def validate_scores(result, labels, expected_cv=None):
    """核验划分、网格、OOF 预测与分数，拒绝串用协议或损坏检查点。"""
    cv = result["cv_details"]
    if (cv["cv_protocol"] != fbcsp.FBCSP_PROTOCOL or cv["scoring"] != "accuracy"
            or cv["shuffle"] is not False or cv["pipeline_defaults"] != fbcsp.PIPELINE_DEFAULTS):
        raise ValueError("FBCSP CV protocol mismatch")
    fbcsp._validate_cv(cv["outer_k_fold"], cv["inner_k_fold"])
    if cv["classes"] != np.unique(labels).tolist():
        raise ValueError("Class order mismatch")
    grid = fbcsp._validate_grid(cv["param_grid"])
    if len(ParameterGrid(grid)) != cv["n_candidates"]:
        raise ValueError("Candidate count mismatch")
    if expected_cv:
        for key in ("outer_k_fold", "inner_k_fold", "param_grid", "pipeline_defaults", "cv_protocol"):
            if cv[key] != expected_cv[key]:
                raise ValueError(f"Permutation CV mismatch: {key}")
    scores = np.asarray(result["fold_scores"], dtype=float)
    splits = list(StratifiedKFold(cv["outer_k_fold"], shuffle=False).split(np.zeros(len(labels)), labels))
    if scores.shape != (len(splits),) or len(cv["folds"]) != len(splits) or not np.isfinite(scores).all():
        raise ValueError("Fold count/score mismatch")
    for number, ((train, test), fold, score) in enumerate(zip(splits, cv["folds"], scores), 1):
        pred = np.asarray(fold["test_predictions"])
        decisions = np.asarray(fold["decision_function"])
        if (fold["fold"] != number or fold["train_indices"] != train.tolist()
                or fold["test_indices"] != test.tolist()
                or fold["test_labels"] != labels[test].tolist()
                or pred.shape != test.shape or not np.isin(pred, np.unique(labels)).all()
                or decisions.shape != test.shape or not np.isfinite(decisions).all()
                or fold["best_params"] not in list(ParameterGrid(grid))
                or not 0 <= fold["best_inner_score"] <= 1
                or not np.isclose(score, np.mean(pred == labels[test]), rtol=0, atol=1e-12)
                or fold["outer_score"] != score):
            raise ValueError("Fold identity/prediction mismatch")
    if not np.isclose(result["accuracy"], scores.mean(), rtol=0, atol=1e-12):
        raise ValueError("Observed accuracy mismatch")


def load_observed(path):
    path = Path(path).resolve()
    digest = fbcsp.file_sha256(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("model") != "FBCSP_SVM" or result.get("status") != "completed":
        raise ValueError("Expected completed FBCSP_SVM result")
    if result["runtime_versions"] != versions():
        raise ValueError("Runtime versions differ; reproduce observed experiment first")
    report, dependencies = load_publication(result["stage4_report"])
    if dependencies != result["dependencies"]:
        raise ValueError("Observed publication identity mismatch")
    data, labels, metadata = fbcsp.load_fbcsp_data(report["ws_output"])
    if metadata != result["input"]:
        raise ValueError("Observed input identity mismatch")
    validate_scores(result, labels)
    dependencies[str(path)] = digest
    verify_dependencies(dependencies)
    return result, data, labels, dependencies


def run_permutation_test(observed_path, *, n_permutations=100, random_seed=42,
                         output_dir=None, resume_from=None, n_jobs=1, native_threads=2):
    integer(n_permutations, "n_permutations")
    integer(random_seed, "random_seed", 0)
    integer(native_threads, "native_threads")
    if type(n_jobs) is not int or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer")
    if output_dir is not None and resume_from is not None:
        raise ValueError("Cannot combine output_dir and resume_from")
    observed, data, labels, dependencies = load_observed(observed_path)
    identity = {"protocol": "permutation_fbcsp_v2_grouped_concat", "dependencies": dependencies,
                "runtime_versions": versions(), "n_permutations": n_permutations,
                "random_seed": random_seed, "observed_accuracy": observed["accuracy"],
                "permutation_unit": "epoch_global", "split_policy": "restratify_permuted_labels",
                "cv": observed["cv_details"]}
    if resume_from:
        checkpoint = Path(resume_from).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        directory = Path(output_dir or config.PROJECT_ROOT / "FBCSP_result" / "permutation" / stamp).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        checkpoint = directory / "permutation_results.json"
    lock = checkpoint.with_suffix(".lock")
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        rng = np.random.default_rng(random_seed)
        record = {"identity": identity, "status": "running", "permutations": [],
                  "n_completed": 0, "p_value": None, "n_extreme": None}
        if resume_from:
            record = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (record["identity"] != identity or record["n_completed"] != len(record["permutations"])
                    or not 0 <= record["n_completed"] <= n_permutations
                    or record["status"] not in ("running", "failed", "completed")):
                raise ValueError("Resume identity/count mismatch")
            for index, item in enumerate(record["permutations"], 1):
                indices = rng.permutation(len(labels))
                permuted = labels[indices]
                if (item["iteration"] != index or item["permutation_indices"] != indices.tolist()
                        or item["permuted_labels"] != permuted.tolist()):
                    raise ValueError("Resume random sequence mismatch")
                validate_scores(item, permuted, observed["cv_details"])
            if record["status"] == "completed":
                p, count = empirical_p_value(observed["accuracy"], [r["accuracy"] for r in record["permutations"]])
                if record["n_completed"] != n_permutations or (record["p_value"], record["n_extreme"]) != (p, count):
                    raise ValueError("Resume final p value mismatch")
                return record
        try:
            record.update(status="running", p_value=None, n_extreme=None)
            record.pop("error", None)
            save_json(checkpoint, record)
            cv = observed["cv_details"]
            for iteration in range(record["n_completed"] + 1, n_permutations + 1):
                indices = rng.permutation(len(labels))
                permuted = labels[indices]
                accuracy, scores, details = fbcsp.nested_cv(
                    data, permuted, k_fold=cv["outer_k_fold"], inner_k_fold=cv["inner_k_fold"],
                    param_grid=cv["param_grid"], n_jobs=n_jobs, native_threads=native_threads,
                    return_details=True)
                verify_dependencies(dependencies)
                item = {"iteration": iteration, "permutation_indices": indices.tolist(),
                        "permuted_labels": permuted.tolist(), "accuracy": accuracy,
                        "fold_scores": scores, "cv_details": details}
                validate_scores(item, permuted, cv)
                record["permutations"].append(item)
                record["n_completed"] = iteration
                save_json(checkpoint, record)
            p, count = empirical_p_value(observed["accuracy"], [r["accuracy"] for r in record["permutations"]])
            record.update(status="completed", p_value=p, n_extreme=count)
            save_json(checkpoint, record)
        except BaseException as exc:
            record.update(status="failed", p_value=None, n_extreme=None,
                          error=f"{type(exc).__name__}: {exc}")
            save_json(checkpoint, record)
            raise
        print(f"FBCSP permutation p={record['p_value']:.6g}; checkpoint={checkpoint}")
        return record
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True, help="classification_result.json")
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--native-threads", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        observed, _, _, _ = load_observed(args.observed)
        print(f"Valid observed FBCSP accuracy: {observed['accuracy']}")
        return observed
    return run_permutation_test(args.observed, n_permutations=args.n_permutations,
                                random_seed=args.seed, output_dir=args.output_dir,
                                resume_from=args.resume, n_jobs=args.n_jobs,
                                native_threads=args.native_threads)


if __name__ == "__main__":
    main()