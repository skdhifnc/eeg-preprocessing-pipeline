"""多频带 CSP-SVM 分类组件。

本模块只负责分类阶段：输入必须是阶段 4 生成的
``(epochs, bands, channels, times)`` 四维数据。固定频带滤波和 Epoch
导出由 :mod:`fbcsp_stage4` 完成；每个频带的 CSP 只在当前训练折内拟合。
本阶段不包含 mRMR。
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
from mne.utils import use_log_level

import config
import ws


FBCSP_PROTOCOL = "nested_fbcsp_v2_grouped_concat"
DEFAULT_OUTER_K_FOLD = 10
DEFAULT_INNER_K_FOLD = 5
FBCSP_PARAM_GRID = {key.replace("csp__", "fbcsp__"): list(values)
                    for key, values in ws.DEFAULT_PARAM_GRID.items()}
PIPELINE_DEFAULTS = {
    "fbcsp__reg": "oas",
    "fbcsp__n_components": 2,
    "fbcsp__log": False,
    "fbcsp__norm_trace": False,
    "svm__kernel": "linear",
    "svm__C": 1.0,
}


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def file_sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _NativeFloat64CSP(CSP):
    """将每带输入规范为 float64，避免 MNE/NumPy 类型差异影响拟合。"""

    def fit(self, X, y):
        if X.dtype is not np.dtype(np.float64):
            X = X.astype(np.float64, copy=True)
        return super().fit(X, y)


class FBCSPTransformer(BaseEstimator, TransformerMixin):
    """对每个频带独立拟合 CSP，并沿特征维拼接各带功率特征。"""

    def __init__(self, reg="oas", n_components=2, log=False, norm_trace=False):
        self.reg = reg
        self.n_components = n_components
        self.log = log
        self.norm_trace = norm_trace

    def _check_X(self, X):
        X = np.asarray(X)
        if X.ndim != 4 or min(X.shape) <= 0:
            raise ValueError(
                "FBCSP input must be non-empty (epochs,bands,channels,times)"
            )
        if X.dtype.kind not in "fiu" or not np.isfinite(X).all():
            raise ValueError("FBCSP input must contain finite numeric values")
        if type(self.n_components) is not int or self.n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if self.n_components > X.shape[2]:
            raise ValueError("n_components cannot exceed the channel count")
        if type(self.log) is not bool or type(self.norm_trace) is not bool:
            raise TypeError("log and norm_trace must be bool values")
        return X.astype(np.float64, copy=False)

    def fit(self, X, y):
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("labels must be a 1D vector matching the epoch count")
        if np.unique(y).size != 2:
            raise ValueError("FBCSP currently supports exactly two classes")
        self.n_bands_in_ = X.shape[1]
        self.n_channels_in_ = X.shape[2]
        self.__dict__.pop("band_csps_", None)
        fitted = []
        for band_index in range(self.n_bands_in_):
            estimator = _NativeFloat64CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=self.log,
                norm_trace=self.norm_trace,
                cov_est="concat",
            )
            estimator.fit(X[:, band_index, :, :], y)
            fitted.append(estimator)
        self.band_csps_ = fitted
        self.feature_metadata_ = [
            {"feature_index": band * self.n_components + component,
             "band_index": band, "csp_component": component}
            for band in range(self.n_bands_in_)
            for component in range(self.n_components)
        ]
        return self

    def transform(self, X):
        check_is_fitted(self, "band_csps_")
        X = self._check_X(X)
        if X.shape[1] != self.n_bands_in_ or X.shape[2] != self.n_channels_in_:
            raise ValueError("FBCSP transform shape does not match fitted data")
        transformed = [
            estimator.transform(X[:, band_index, :, :])
            for band_index, estimator in enumerate(self.band_csps_)
        ]
        result = np.concatenate(transformed, axis=1).astype(np.float64, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("FBCSP transform produced non-finite features")
        return result


def _pipeline():
    return Pipeline([
        ("fbcsp", FBCSPTransformer(
            reg="oas", n_components=2, log=False, norm_trace=False
        )),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0)),
    ])


def validate_fbcsp_hdf5(hdf5_path=None):
    """校验阶段 4 FBCSP 四维 HDF5 契约。"""
    path = Path(config.FBCSP_WS_OUTPUT_PATH if hdf5_path is None else hdf5_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FBCSP HDF5 not found: {path}")
    with h5py.File(path, "r") as h5_file:
        required = ("clean_eeg", "labels", "channel_names", "times", "event_samples", "bands")
        for name in required:
            if name not in h5_file or not isinstance(h5_file[name], h5py.Dataset):
                raise ValueError(f"Missing FBCSP HDF5 dataset: {name}")
        data = h5_file["clean_eeg"]
        if data.ndim != 4 or min(data.shape) <= 0:
            raise ValueError(f"clean_eeg must be non-empty 4D, got {data.shape}")
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError("FBCSP clean_eeg must have a floating dtype")
        for start in range(0, data.shape[0], 8):
            if not np.isfinite(data[start:start + 8]).all():
                raise ValueError("FBCSP clean_eeg contains NaN or infinite values")
        labels = h5_file["labels"][:]
        samples = h5_file["event_samples"][:]
        times = h5_file["times"][:]
        bands = h5_file["bands"][:]
        if labels.shape != (data.shape[0],) or samples.shape != labels.shape:
            raise ValueError("labels and event_samples must match the epoch count")
        if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(samples.dtype, np.integer):
            raise ValueError("labels and event_samples must be integer vectors")
        if np.unique(labels).size != 2:
            raise ValueError("FBCSP input must contain exactly two classes")
        if times.shape != (data.shape[3],) or bands.shape != (data.shape[1], 2):
            raise ValueError("FBCSP times/bands dimensions do not match clean_eeg")
        if np.any(samples < 0) or np.any(samples[1:] <= samples[:-1]):
            raise ValueError("event_samples must be strictly increasing")
        if not np.isfinite(times).all() or not np.isfinite(bands).all():
            raise ValueError("times and bands must be finite")
        if np.any(bands[:, 0] <= 0) or np.any(bands[:, 0] >= bands[:, 1]):
            raise ValueError("bands must satisfy 0 < low < high")
        if _text(h5_file.attrs.get("axis_order", "")) != "epochs,bands,channels,times":
            raise ValueError("Unexpected FBCSP axis_order")
        if h5_file.attrs.get("stage") != 4:
            raise ValueError("FBCSP input must declare stage=4")
        for key in ("run_id", "source_raw_sha256", "sampling_rate", "unit", "tmin", "tmax", "label_map"):
            if key not in h5_file.attrs:
                raise ValueError(f"Missing FBCSP attribute: {key}")
        sfreq = h5_file.attrs["sampling_rate"]
        if (isinstance(sfreq, (bool, np.bool_)) or not isinstance(sfreq, (int, float, np.integer, np.floating))
                or not np.isfinite(sfreq) or sfreq <= 0):
            raise ValueError("Invalid sampling_rate")
        if np.any(bands[:, 1] >= sfreq / 2) or len(np.unique(bands, axis=0)) != len(bands):
            raise ValueError("Invalid or duplicate frequency bands")
        if np.any(bands[1:, 0] < bands[:-1, 1]):
            raise ValueError("Frequency bands must be ordered and non-overlapping")
        if not np.allclose(np.diff(times), 1 / sfreq, rtol=0, atol=1e-9):
            raise ValueError("times do not match sampling_rate")
        if not np.allclose(times[[0, -1]], [h5_file.attrs["tmin"], h5_file.attrs["tmax"]], rtol=0, atol=1e-9):
            raise ValueError("tmin/tmax do not match times")
        channels = h5_file["channel_names"]
        if channels.shape != (data.shape[2],) or h5py.check_string_dtype(channels.dtype) is None:
            raise ValueError("Invalid channel_names")
        names = [_text(v).strip() for v in channels[:]]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("channel_names must be unique and non-empty")
        if _text(h5_file.attrs["unit"]) not in ("volts", "millivolts", "microvolts", "nanovolts"):
            raise ValueError("Invalid unit")
        label_map = json.loads(_text(h5_file.attrs["label_map"]))
        if (not isinstance(label_map, dict) or len(label_map) != 2
                or any(type(v) is not int for v in label_map.values())
                or set(label_map.values()) != set(labels.tolist())):
            raise ValueError("label_map does not match labels")
        digest = _text(h5_file.attrs["source_raw_sha256"])
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid source_raw_sha256")
        if not _text(h5_file.attrs["run_id"]).strip():
            raise ValueError("Empty run_id")
    return path


def load_fbcsp_data(hdf5_path=None):
    path = validate_fbcsp_hdf5(hdf5_path)
    before = file_sha256(path)
    with h5py.File(path, "r") as h5_file:
        data = h5_file["clean_eeg"][:].astype(np.float64)
        labels = h5_file["labels"][:]
        bands = h5_file["bands"][:].astype(np.float64)
        metadata = {
            "path": str(path),
            "input_sha256": before,
            "run_id": _text(h5_file.attrs["run_id"]),
            "source_raw_sha256": _text(h5_file.attrs["source_raw_sha256"]),
            "sampling_rate": float(h5_file.attrs["sampling_rate"]),
            "unit": _text(h5_file.attrs["unit"]),
            "axis_order": _text(h5_file.attrs["axis_order"]),
            "bands": bands.tolist(),
            "shape": list(data.shape),
        }
    if before != file_sha256(path) or not np.isfinite(data).all():
        raise RuntimeError("FBCSP input changed or overflowed while loading")
    data.setflags(write=False)
    labels.setflags(write=False)
    return data, labels, metadata


def _validate_grid(param_grid, n_channels=32):
    source = FBCSP_PARAM_GRID if param_grid is None else param_grid
    if not isinstance(source, dict) or not source:
        raise ValueError("param_grid must be a non-empty dictionary")
    unknown = set(source) - set(FBCSP_PARAM_GRID)
    if unknown:
        raise ValueError(f"Unsupported FBCSP parameter keys: {sorted(unknown)}")
    # 复用 baseline 的数值和类型验证；默认候选也直接由 baseline 派生。
    translated = None if param_grid is None else {
        key.replace("fbcsp__", "csp__"): values for key, values in source.items()}
    grid = ws._search_grid(translated, n_channels)
    return {key.replace("csp__", "fbcsp__"): values for key, values in grid.items()}


def _validate_cv(k_fold, inner_k_fold):
    for value, name in ((k_fold, "k_fold"), (inner_k_fold, "inner_k_fold")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < 2:
            raise ValueError(f"{name} must be at least 2")


def _grouped_features(band_csps, X_train, X_val, n_components, log):
    """逐带先截取滤波器再投影；全部特征在 Scaler/SVM 拟合之前生成。

    log=False 的均值/标准差必须由当前内层训练功率重新计算，不能直接使用
    max_components 拟合时的归一化参数，也不能在拼接后截取前若干列。
    """
    train_features, val_features = [], []
    for band, csp in enumerate(band_csps):
        filters = csp.filters_[:n_components]
        train_power = ws._mean_power(filters, X_train[:, band])
        val_power = ws._mean_power(filters, X_val[:, band])
        if log:
            train_feature, val_feature = np.log(train_power), np.log(val_power)
        else:
            mean, std = train_power.mean(axis=0), train_power.std(axis=0)
            train_feature = (train_power - mean) / std
            val_feature = (val_power - mean) / std
        train_features.append(train_feature)
        val_features.append(val_feature)
    return np.concatenate(train_features, axis=1), np.concatenate(val_features, axis=1)


def _grouped_inner_fold(data, labels, train, validation, candidates, native_threads):
    """局部索引仅作用于外层训练集；每个 reg/频带只拟合内层训练 epoch。"""
    with threadpool_limits(limits=native_threads), use_log_level("WARNING"):
        X_train, y_train = data[train], labels[train]
        X_val, y_val = data[validation], labels[validation]
        resolved = [{**PIPELINE_DEFAULTS, **candidate} for candidate in candidates]
        reg_groups = {}
        for index, params in enumerate(resolved):
            reg_groups.setdefault(params["fbcsp__reg"], []).append(index)
        scores = np.full(len(candidates), np.nan)
        for reg, indices in reg_groups.items():
            max_components = max(resolved[i]["fbcsp__n_components"] for i in indices)
            transformer = FBCSPTransformer(reg=reg, n_components=max_components, log=False)
            transformer.fit(X_train, y_train)
            for count in dict.fromkeys(resolved[i]["fbcsp__n_components"] for i in indices):
                matching = [i for i in indices if resolved[i]["fbcsp__n_components"] == count]
                for log in dict.fromkeys(resolved[i]["fbcsp__log"] for i in matching):
                    train_feature, val_feature = _grouped_features(
                        transformer.band_csps_, X_train, X_val, count, log)
                    scaler = StandardScaler()
                    train_feature = scaler.fit_transform(train_feature)
                    val_feature = scaler.transform(val_feature)
                    for i in matching:
                        if resolved[i]["fbcsp__log"] == log:
                            svm = SVC(kernel="linear", C=resolved[i]["svm__C"])
                            svm.fit(train_feature, y_train)
                            scores[i] = float(svm.score(val_feature, y_val))
        return scores


def grouped_search(data, labels, grid, *, inner_k_fold=5, n_jobs=1, native_threads=2):
    """concat/average_power/mutual_info、norm_trace=False 的分组内层搜索。

    不存储跨折或跨置换缓存。保留 ParameterGrid 候选顺序、折分数等权平均和
    并列首选规则；最终在整个外层训练集上重新拟合正式 Pipeline。
    """
    grid = _validate_grid(grid, data.shape[2])
    candidates = list(ParameterGrid(grid))
    splits = list(StratifiedKFold(inner_k_fold, shuffle=False).split(data, labels))
    with threadpool_limits(limits=native_threads), use_log_level("WARNING"), parallel_config(
            backend="loky", inner_max_num_threads=native_threads,
            initializer=ws._quiet_search_worker):
        columns = Parallel(n_jobs=n_jobs, pre_dispatch="n_jobs")(
            delayed(_grouped_inner_fold)(data, labels, train, validation, candidates, native_threads)
            for train, validation in splits)
        scores = np.asarray(columns).T
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite grouped candidate scores")
        mean_scores = np.average(scores, axis=1)
        best_index = int(np.argmax(mean_scores))
        model = _pipeline().set_params(**candidates[best_index])
        model.fit(data, labels)
    return {"params": candidates, "split_scores": scores, "mean_scores": mean_scores,
            "best_index": best_index, "best_estimator": model}


def nested_cv(data, labels, *, k_fold=DEFAULT_OUTER_K_FOLD,
              inner_k_fold=DEFAULT_INNER_K_FOLD, param_grid=None,
              n_jobs=1, return_details=False, cache_transformers=False,
              native_threads=2, baseline=False, search_engine="grouped"):
    """默认分组搜索；reference 提供无缓存 GridSearchCV 数值对照。

    cache_transformers 仅为兼容旧调用保留，不再启用任何变换磁盘缓存。
    """
    ws._validate_cv_options(k_fold, inner_k_fold, n_jobs, return_details)
    if type(native_threads) is not int or native_threads < 1:
        raise ValueError("native_threads must be a positive integer")
    if type(cache_transformers) is not bool or type(baseline) is not bool:
        raise TypeError("cache_transformers and baseline must be bool")
    if search_engine not in ("grouped", "reference"):
        raise ValueError("search_engine must be grouped or reference")
    if not isinstance(data, np.ndarray) or data.ndim != 4:
        raise ValueError("data must be a 4D NumPy array")
    if min(data.shape) <= 0 or data.dtype.kind not in "fiu" or not np.isfinite(data).all():
        raise ValueError("data must be non-empty finite real EEG")
    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != data.shape[0]:
        raise ValueError("labels must match data epoch count")
    values, counts = np.unique(labels, return_counts=True)
    if labels.dtype.kind not in "iu" or len(values) != 2 or k_fold > counts.min():
        raise ValueError("Need two integer classes with at least k_fold samples each")
    if baseline and data.shape[1] != 1:
        raise ValueError("baseline mode requires one broad frequency band")
    grid = _validate_grid(param_grid, data.shape[2])
    search_grid = ({k.replace("fbcsp__", "csp__"): v for k, v in grid.items()}
                   if baseline else grid)
    outer_cv = StratifiedKFold(n_splits=k_fold, shuffle=False)
    outer_splits = list(outer_cv.split(data, labels))
    for train_idx, _ in outer_splits:
        if inner_k_fold > np.unique(labels[train_idx], return_counts=True)[1].min():
            raise ValueError("inner_k_fold exceeds training class count")
    scores = []
    details = {
        "cv_method": "nested_stratified_k_fold",
        "outer_k_fold": int(k_fold), "inner_k_fold": int(inner_k_fold),
        "shuffle": False, "scoring": "accuracy", "n_jobs": int(n_jobs),
        "param_grid": search_grid, "n_candidates": len(ParameterGrid(grid)),
        "pipeline_defaults": dict(ws.PIPELINE_DEFAULTS if baseline else PIPELINE_DEFAULTS),
        "cv_protocol": "nested_csp_paired_v2_grouped_concat" if baseline else FBCSP_PROTOCOL, "folds": [],
        "classes": values.tolist(),
    }
    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        train_labels = labels[train_idx]
        with threadpool_limits(limits=native_threads), use_log_level("WARNING"), \
                parallel_config(backend="loky", inner_max_num_threads=native_threads,
                                initializer=ws._quiet_search_worker):
            X = data[:, 0] if baseline else data
            if search_engine == "grouped":
                search = ws.grouped_search if baseline else grouped_search
                result = search(X[train_idx], train_labels, search_grid,
                                inner_k_fold=inner_k_fold, n_jobs=n_jobs,
                                native_threads=native_threads)
                best = result["best_index"]
                best_params = result["params"][best]
                best_score = result["mean_scores"][best]
                model = result["best_estimator"]
            else:
                search = GridSearchCV(
                    ws._pipeline() if baseline else _pipeline(), search_grid, scoring="accuracy",
                    cv=StratifiedKFold(n_splits=inner_k_fold, shuffle=False),
                    refit=True, n_jobs=n_jobs, pre_dispatch="n_jobs", error_score="raise",
                    return_train_score=False,
                ).fit(X[train_idx], train_labels)
                best_params, best_score = search.best_params_, search.best_score_
                model = search.best_estimator_
            prediction = model.predict(X[test_idx])
            decision = model.decision_function(X[test_idx])
        score = float(np.mean(prediction == labels[test_idx]))
        scores.append(score)
        details["folds"].append({
            "fold": fold,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "best_params": dict(best_params),
            "best_inner_score": float(best_score),
            "outer_score": score,
            "test_labels": labels[test_idx].tolist(),
            "test_predictions": prediction.tolist(),
            "decision_function": decision.tolist(),
        })
        if not baseline:
            details["folds"][-1]["feature_metadata"] = model["fbcsp"].feature_metadata_
        print(f"{'CSP' if baseline else 'FBCSP'} fold {fold}/{k_fold}: {score:.4f}; {best_params}", flush=True)
    average = float(np.mean(scores))
    if return_details:
        return average, scores, details
    return average, scores


def metrics(details):
    """按原试次索引恢复 OOF 预测；AUC 不跨折混合未校准 decision 分值。"""
    rows = sorted((i, y, pred) for fold in details["folds"]
                  for i, y, pred in zip(fold["test_indices"], fold["test_labels"], fold["test_predictions"]))
    y, pred = np.asarray([r[1] for r in rows]), np.asarray([r[2] for r in rows])
    matrix = confusion_matrix(y, pred, labels=details["classes"])
    return {"oof_accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "confusion_matrix": matrix.tolist(),
            "class_recall": {str(c): float(matrix[i, i] / matrix[i].sum())
                             for i, c in enumerate(details["classes"])},
            "oof_labels": y.tolist(), "oof_predictions": pred.tolist()}
