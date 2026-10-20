# 分类模块 - 公共空间模式(CSP)+支持向量机(SVM)进行BCI运动想象分类
import json
import hashlib
from pathlib import Path
from contextlib import nullcontext
import tempfile
from time import perf_counter
from types import SimpleNamespace

import config
import h5py  # HDF5文件处理
import numpy as np  # 数值计算
from mne.decoding import CSP  # MNE库提供的公共空间模式算法
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # 特征标准化
from sklearn.svm import SVC  # 支持向量机分类器
from mne.filter import filter_data
from mne.utils import use_log_level
from joblib import Memory, Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits


DEFAULT_INNER_K_FOLD = 5
CV_PROTOCOL = "nested_csp_v1"
PIPELINE_DEFAULTS = {"csp__n_components": 2, "csp__reg": "oas",
                     "csp__log": False, "csp__norm_trace": False,
                     "svm__kernel": "linear", "svm__C": 1.0}
DEFAULT_PARAM_GRID = {
    "csp__reg": [0.1,"ledoit_wolf",],
    "csp__n_components": [2, 4],
    "csp__log": [True],
    "svm__C": [0.1, 1.0, 10.0],
}


def _validate_fold_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 2:
        raise ValueError(f"{name} must be at least 2")


def file_sha256(path):
    """流式计算输入/结果身份，不加载完整文件到内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions():
    import mne
    import sklearn
    return {"numpy": np.__version__, "mne": mne.__version__, "sklearn": sklearn.__version__}


def _quiet_search_worker():
    """loky 子进程不会继承父进程的 MNE 日志上下文。"""
    import mne
    mne.set_log_level("WARNING")


class _NativeFloat64CSP(CSP):
    """规范化进程反序列化后的 NumPy dtype，兼容 MNE 1.8 的 copy=None 检查。

    numpy2/loky 反序列化可能保留非规范 float64 dtype 对象。数值不变，
    仅必要时复制；不改变 CSP 算法、参数或训练样本。
    """

    def fit(self, X, y):
        if X.dtype is not np.dtype(np.float64):
            X = X.astype(np.float64, copy=True)
        return super().fit(X, y)


def _validate_cv_options(k_fold, inner_k_fold, n_jobs, return_details):
    _validate_fold_count(k_fold, "k_fold")
    _validate_fold_count(inner_k_fold, "inner_k_fold")
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)):
        raise TypeError("n_jobs must be a non-zero integer")
    if n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer")
    if type(return_details) is not bool:
        raise TypeError("return_details must be True or False")


def _search_grid(param_grid, n_channels):
    """生成独立网格并在训练前校验；不依据全数据拟合结果选择候选值。"""
    source = DEFAULT_PARAM_GRID if param_grid is None else param_grid
    if not isinstance(source, dict) or not source:
        raise ValueError("param_grid must be a non-empty dictionary")
    unknown = set(source) - set(DEFAULT_PARAM_GRID)
    if unknown:
        raise ValueError(f"Unsupported parameter keys: {sorted(unknown)}")
    # ParameterGrid 同时校验候选必须是非空序列。缺省维度沿用 Pipeline 的固定值。
    ParameterGrid(source)
    grid = {key: [value.item() if isinstance(value, np.generic) else value for value in values]
            for key, values in source.items()}
    if param_grid is None:
        grid["csp__n_components"] = [n for n in grid["csp__n_components"] if n <= n_channels]
    for key, values in grid.items():
        for value in values:
            if key == "csp__reg":
                valid = (value in ("ledoit_wolf", "oas") if isinstance(value, str)
                         else type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1)
            elif key == "csp__n_components":
                valid = type(value) is int and 1 <= value <= n_channels
            elif key == "csp__log":
                valid = type(value) is bool
            else:
                valid = type(value) in (int, float) and np.isfinite(value) and value > 0
            if not valid:
                raise ValueError(f"Invalid candidate for {key}: {value!r}")
    return grid


def _pipeline():
    return Pipeline([
        ("csp", _NativeFloat64CSP(n_components=2, reg="oas", log=False, norm_trace=False)),
        ("scaler", StandardScaler()), ("svm", SVC(kernel="linear", C=1.0)),
    ])


def _mean_power(filters, data):
    # 严格沿用 MNE 1.8 的每 epoch dot/平方/时间均值运算顺序。
    # 不先投影4成分后裁为2成分，避免不同矩阵尺寸改变 BLAS 浮点路径。
    projected = np.asarray([np.dot(filters, epoch) for epoch in data])
    return (projected ** 2).mean(axis=2)


def _grouped_inner_fold(data, labels, train, validation, candidates, native_threads):
    """一个任务只处理一个内层折，标签和 EEG 都局限于该外层训练集。

    任一 reg 的空间滤波器只从 train 估计。验证标签只用于 score；没有磁盘缓存，
    特征数组和模型在函数返回后释放，返回值只有按候选顺序排列的分数。
    """
    with threadpool_limits(limits=native_threads), use_log_level("WARNING"):
        X_train, y_train = data[train], labels[train]
        X_val, y_val = data[validation], labels[validation]
        resolved = [{**PIPELINE_DEFAULTS, **candidate} for candidate in candidates]
        reg_groups = {}
        for index, params in enumerate(resolved):
            reg_groups.setdefault(params["csp__reg"], []).append(index)
        scores = np.empty(len(candidates))
        for reg, indices in reg_groups.items():
            max_components = max(resolved[i]["csp__n_components"] for i in indices)
            csp = _NativeFloat64CSP(n_components=max_components, reg=reg,
                                    log=False, norm_trace=False)
            csp.fit(X_train, y_train)
            for n_components in dict.fromkeys(resolved[i]["csp__n_components"] for i in indices):
                train_power = _mean_power(csp.filters_[:n_components], X_train)
                val_power = _mean_power(csp.filters_[:n_components], X_val)
                matching = [i for i in indices if resolved[i]["csp__n_components"] == n_components]
                for log in dict.fromkeys(resolved[i]["csp__log"] for i in matching):
                    if log:
                        train_feature, val_feature = np.log(train_power), np.log(val_power)
                    else:
                        # MNE CSP(log=False) 已先执行训练均值/标准差归一化，
                        # 后面仍需保留原 Pipeline 的 StandardScaler，不能省略。
                        mean, std = train_power.mean(axis=0), train_power.std(axis=0)
                        train_feature = (train_power - mean) / std
                        val_feature = (val_power - mean) / std
                    scaler = StandardScaler()
                    train_feature = scaler.fit_transform(train_feature)
                    val_feature = scaler.transform(val_feature)
                    for i in matching:
                        if resolved[i]["csp__log"] == log:
                            svm = SVC(kernel="linear", C=resolved[i]["svm__C"])
                            svm.fit(train_feature, y_train)
                            scores[i] = float(svm.score(val_feature, y_val))
        return scores


def grouped_search(data, labels, grid, *, inner_k_fold=5, n_jobs=1, native_threads=2):
    """等价于当前受限网格的 GridSearchCV；完整候选/折分数供审计与对照。

    仅适用于当前 CSP average_power/concat/mutual_info/norm_trace=False 与线性SVM。
    _search_grid 拒绝扩展到未验证参数。维持 ParameterGrid 顺序与并列首选规则。
    """
    grid = _search_grid(grid, data.shape[1])
    candidates = list(ParameterGrid(grid))
    splits = list(StratifiedKFold(inner_k_fold, shuffle=False).split(data, labels))
    with threadpool_limits(limits=native_threads), use_log_level("WARNING"), parallel_config(
            backend="loky", inner_max_num_threads=native_threads, initializer=_quiet_search_worker):
        # 仅内层折并行；每任务独立一次 train/validation 切片，不按72候选反复切片。
        columns = Parallel(n_jobs=n_jobs, pre_dispatch="n_jobs")(
            delayed(_grouped_inner_fold)(data, labels, train, validation, candidates, native_threads)
            for train, validation in splits)
        scores = np.asarray(columns).T
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite grouped candidate scores")
        # GridSearchCV 对各折准确率取等权算术平均，不能按验证样本数加权。
        mean_scores = np.average(scores, axis=1)
        best_index = int(np.argmax(mean_scores))  # 并列时原 ParameterGrid 中首项
        model = _pipeline().set_params(**candidates[best_index])
        # 完整外层训练集重新拟合原 MNE Pipeline；不使用内层模型评分外层测试集。
        model.fit(data, labels)
    return {"params": candidates, "split_scores": scores, "mean_scores": mean_scores,
            "best_index": best_index, "best_estimator": model}


def _input_path(hdf5_path):
    """默认读取阶段 4 分类窗口；不依赖终端当前目录。"""
    return Path(config.STAGE4_WS_OUTPUT_PATH if hdf5_path is None else hdf5_path).expanduser().resolve()


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def validate_clean_hdf5(hdf5_path=None):
    """验证 clean HDF5；声明 stage=4 时额外核对阶段 4 元数据。

    同时供阶段 4 发布 full/ws 文件使用，故不限制窗口或每类最小试次数。
    成功返回 None。这里只校验文件内部契约，不代替上游 ICA 凭据/哈希审核。
    """
    path = _input_path(hdf5_path)
    if not path.is_file():
        raise FileNotFoundError(f"Clean HDF5 not found: {path}; run stage 4 first")
    with h5py.File(path, "r") as h5_file:
        required = ["clean_eeg", "labels", "channel_names"]
        stage = h5_file.attrs.get("stage")
        if stage is not None and (np.ndim(stage) != 0 or stage != 4):
            raise ValueError("Unsupported HDF5 stage; expected stage=4")
        if stage is not None:
            required += ["times", "event_samples"]
        for name in required:
            if name not in h5_file or not isinstance(h5_file[name], h5py.Dataset):
                raise ValueError(f"Missing HDF5 dataset: {name}")

        data, labels = h5_file["clean_eeg"], h5_file["labels"]
        if data.ndim != 3 or min(data.shape) <= 0:
            raise ValueError(f"clean_eeg must be non-empty 3D, got {data.shape}")
        if labels.shape != (data.shape[0],):
            raise ValueError("labels must be 1D and match the epoch count")
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError("clean_eeg must have a floating dtype")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must have an integer dtype")
        for start in range(0, data.shape[0], 64):
            if not np.isfinite(data[start:start + 64]).all():
                raise ValueError("clean_eeg contains NaN or infinite values")
        unique_labels = np.unique(labels[:])
        if len(unique_labels) != 2:
            raise ValueError("Expected exactly two classes in labels")

        channels = h5_file["channel_names"]
        if channels.shape != (data.shape[1],) or h5py.check_string_dtype(channels.dtype) is None:
            raise ValueError("channel_names must be a string vector matching the channel count")
        names = [_text(name).strip() for name in channels[:]]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("channel_names must be unique and non-empty")
        sfreq = h5_file.attrs.get("sampling_rate")
        if (np.ndim(sfreq) != 0 or isinstance(sfreq, (bool, np.bool_))
                or not isinstance(sfreq, (int, float, np.integer, np.floating))
                or not np.isfinite(sfreq) or sfreq <= 0):
            raise ValueError("sampling_rate must be a positive finite number")
        axis_order = h5_file.attrs.get("axis_order", "epochs,channels,times")
        if _text(axis_order) != "epochs,channels,times":
            raise ValueError("axis_order must be epochs,channels,times")
        if stage is None:
            return  # 兼容旧 clean 文件，但不将其当作正式阶段 4 输出。

        for name in ("axis_order", "unit", "label_map", "run_id", "tmin", "tmax"):
            if name not in h5_file.attrs:
                raise ValueError(f"Missing stage 4 attribute: {name}")
        if _text(h5_file.attrs["unit"]) not in {"volts", "millivolts", "microvolts", "nanovolts"}:
            raise ValueError("Unsupported stage 4 unit")
        if not _text(h5_file.attrs["run_id"]).strip():
            raise ValueError("Stage 4 run_id must be non-empty")
        try:
            label_map = json.loads(_text(h5_file.attrs["label_map"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("label_map must be valid JSON") from exc
        if (not isinstance(label_map, dict) or len(label_map) != 2
                or any(not name for name in label_map)
                or any(type(value) is not int for value in label_map.values())
                or set(label_map.values()) != set(unique_labels.tolist())):
            raise ValueError("label_map must define the two integer classes in labels")
        times, samples = h5_file["times"], h5_file["event_samples"]
        if times.shape != (data.shape[2],) or not np.issubdtype(times.dtype, np.floating):
            raise ValueError("times must be a floating vector matching the time dimension")
        times = times[:]
        if (not np.isfinite(times).all()
                or not np.allclose(np.diff(times), 1.0 / sfreq, rtol=1e-6, atol=1e-10)):
            raise ValueError("times spacing must match sampling_rate")
        if not np.allclose([times[0], times[-1]],
                           [h5_file.attrs["tmin"], h5_file.attrs["tmax"]], rtol=0, atol=1e-9):
            raise ValueError("tmin/tmax must match times endpoints")
        if samples.shape != labels.shape or not np.issubdtype(samples.dtype, np.integer):
            raise ValueError("event_samples must be an integer vector matching labels")
        samples = samples[:]
        if np.any(samples < 0) or np.any(samples[1:] <= samples[:-1]):
            raise ValueError("event_samples must be non-negative and strictly increasing")


def load_clean_data(hdf5_path=None):
    """
    加载已经去伪迹处理好的干净EEG数据和标签
    
    参数:
        hdf5_path: HDF5文件路径
        
    返回:
        data: 形状(n_epochs, n_channels, n_times) 干净EEG数据
        labels: 形状(n_epochs,) 每个epoch的标签，保留文件内的整数编码
    """
    hdf5_path = _input_path(hdf5_path)
    validate_clean_hdf5(hdf5_path)  # 先验证文件格式
    with h5py.File(hdf5_path, "r") as h5_file:
        data = h5_file["clean_eeg"][:].astype(np.float64)  # 加载数据并转换为float64，满足MNE-CSP要求
        labels = h5_file["labels"][:]  # 加载标签
    if not np.isfinite(data).all():
        raise ValueError("EEG conversion to float64 overflowed")
    return data, labels


def load_evaluation_data(hdf5_path=None, filter_band=None):
    """只读加载 EEG；可选的固定、无标签滤波仅执行一次，供各次置换共用。"""
    hdf5_path = _input_path(hdf5_path)
    data, labels = load_clean_data(hdf5_path)
    if filter_band is not None:
        with h5py.File(hdf5_path, "r") as h5_file:
            sfreq = float(h5_file.attrs["sampling_rate"])
        band = np.asarray(filter_band)
        if (band.shape != (2,) or band.dtype.kind not in "iuf"
                or not np.isfinite(band).all() or not 0 < band[0] < band[1] < sfreq / 2):
            raise ValueError("filter_band must satisfy 0 < low < high < sampling_rate / 2")
        data = filter_data(data, sfreq, float(band[0]), float(band[1]), verbose=False)
    return data, labels


def train_test(hdf5_path=None, k_fold=10, filter_band=None, *,
               inner_k_fold=DEFAULT_INNER_K_FOLD, param_grid=None, n_jobs=1,
               return_details=False, cache_transformers=False, native_threads=2,
               search_engine="reference"):
    """
    使用 CSP + SVM 进行嵌套分层交叉验证；外层评估，内层选参。
    
    参数:
        hdf5_path: 默认 config.STAGE4_WS_OUTPUT_PATH；也接受兼容的 clean HDF5
        k_fold: 外层折数，默认10折；保持 shuffle=False
        filter_band: 可选额外带通 (低频, 高频)，使用文件采样率；默认不重复滤波
        inner_k_fold: 内层折数，默认5折；只划分当前外层训练集
        param_grid: Pipeline 参数网格字典，None 使用 DEFAULT_PARAM_GRID
        n_jobs: 内层搜索并行数，默认1；-1 使用所有可用CPU，可能增加内存占用
        return_details: True 时额外返回网格、各折最优参数/内层分数/外层索引
        cache_transformers: 缓存同一训练子集/标签/CSP参数的变换，默认False
        native_threads: 每个进程底层数学库线程上限，默认2
        search_engine: reference 原 GridSearchCV；grouped 折内按 reg 分组复用
        
    返回:
        average_score: 外层测试准确率的算术平均，不是内层最佳分数
        scores: 每个外层测试折的准确率，一维 list[float]
        details: 仅 return_details=True 时作为第三项返回，可直接 JSON 序列化

    CSP 和 StandardScaler 必须在 Pipeline 内；禁止在内层划分前预先 fit。
    已发布 HDF5 的 ICA 等上游预处理不在本函数的交叉验证范围内。
    """
    _validate_cv_options(k_fold, inner_k_fold, n_jobs, return_details)
    data, labels = load_evaluation_data(hdf5_path, filter_band)
    return nested_cv(data, labels, k_fold=k_fold, inner_k_fold=inner_k_fold,
                     param_grid=param_grid, n_jobs=n_jobs, return_details=return_details,
                     cache_transformers=cache_transformers, native_threads=native_threads,
                     search_engine=search_engine)


def nested_cv(data, labels, *, k_fold=10, inner_k_fold=DEFAULT_INNER_K_FOLD,
              param_grid=None, n_jobs=1, return_details=False,
              cache_transformers=False, native_threads=2, search_engine="reference"):
    """共享入口：每次重新搜索；可缓存相同训练数据/标签/变换参数的中间结果。

    划分依据本次传入的 labels；用于置换时不是固定原始标签产生的分层折。
    data 为 (epoch, channel, time)，labels 每个 epoch 恰好一个整数标签。
    """
    _validate_cv_options(k_fold, inner_k_fold, n_jobs, return_details)
    if search_engine not in ("reference", "grouped"):
        raise ValueError("search_engine must be reference or grouped")
    if type(cache_transformers) is not bool:
        raise TypeError("cache_transformers must be True or False")
    if (isinstance(native_threads, (bool, np.bool_))
            or not isinstance(native_threads, (int, np.integer)) or native_threads < 1):
        raise ValueError("native_threads must be a positive integer")
    native_threads = int(native_threads)
    data, labels = np.asarray(data, dtype=np.float64), np.asarray(labels)
    if data.ndim != 3 or min(data.shape) <= 0 or not np.isfinite(data).all():
        raise ValueError("data must be finite non-empty 3D EEG (epochs, channels, times)")
    if (labels.shape != (len(data),) or not np.issubdtype(labels.dtype, np.integer)
            or len(np.unique(labels)) != 2):
        raise ValueError("labels must contain one integer label per epoch and exactly two classes")
    max_folds = int(np.unique(labels, return_counts=True)[1].min())
    if k_fold > max_folds:
        raise ValueError(f"k_fold={k_fold} exceeds minimum class count={max_folds}; "
                         f"need 2 <= k_fold <= {max_folds}")
    if data.shape[1] < 2 or data.shape[2] < 2:
        raise ValueError("CSP requires at least two channels and two time samples")
    grid = _search_grid(param_grid, data.shape[1])
    outer_cv = StratifiedKFold(n_splits=k_fold, shuffle=False)
    outer_splits = list(outer_cv.split(data, labels))
    # 提前检查所有外层训练折，避免完成部分搜索后才发现内层样本不足。
    for fold, (train_idx, _) in enumerate(outer_splits, start=1):
        inner_max = int(np.unique(labels[train_idx], return_counts=True)[1].min())
        if inner_k_fold > inner_max:
            raise ValueError(f"Outer fold {fold}: inner_k_fold={inner_k_fold} exceeds "
                             f"training minimum class count={inner_max}; no automatic fold reduction")
    scores = []
    details = {
        "cv_method": "nested_stratified_k_fold",
        "outer_k_fold": int(k_fold), "inner_k_fold": int(inner_k_fold),
        "shuffle": False, "scoring": "accuracy", "n_jobs": int(n_jobs),
        "param_grid": grid, "n_candidates": len(ParameterGrid(grid)),
        "pipeline_defaults": dict(PIPELINE_DEFAULTS),
        "cv_protocol": CV_PROTOCOL,
        "folds": [],
    }
    for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        train_data = data[train_idx]
        train_labels = labels[train_idx]
        started = perf_counter()
        print(f"Outer fold {fold}/{k_fold}: {len(train_idx)} train, {len(test_idx)} test; "
              f"{details['n_candidates']} candidates x {inner_k_fold} inner folds", flush=True)
        # 缓存目录每个外层折独立且及时清理。joblib 将训练 X、y 和变换器参数
        # 一同纳入键；改变 C 可以命中，改变标签/内层训练集/CSP参数不能命中。
        # 不缓存验证/测试分数，不跨外层折或置换共享。
        cache_context = (tempfile.TemporaryDirectory(prefix="bci-csp-")
                         if cache_transformers and search_engine == "reference" else nullcontext(None))
        with cache_context as cache_dir, threadpool_limits(limits=native_threads), \
                use_log_level("WARNING"), parallel_config(
                    backend="loky", inner_max_num_threads=native_threads,
                    initializer=_quiet_search_worker):
            if search_engine == "grouped":
                result = grouped_search(train_data, train_labels, grid, inner_k_fold=inner_k_fold,
                                        n_jobs=n_jobs, native_threads=native_threads)
                best = result["best_index"]
                search = SimpleNamespace(best_params_=result["params"][best],
                                         best_score_=float(result["mean_scores"][best]))
                score = float(result["best_estimator"].score(data[test_idx], labels[test_idx]))
                del result
            else:
                pipeline = _pipeline()
                pipeline.memory = Memory(cache_dir, verbose=0) if cache_dir else None
                search = GridSearchCV(
                    pipeline, grid, scoring="accuracy",
                    cv=StratifiedKFold(n_splits=inner_k_fold, shuffle=False),
                    refit=True, n_jobs=n_jobs, pre_dispatch="n_jobs",
                    error_score="raise", return_train_score=False,
                )
                search.fit(train_data, train_labels)
                score = float(search.score(data[test_idx], labels[test_idx]))
                del pipeline
        scores.append(score)
        details["folds"].append({
            "fold": fold, "train_indices": train_idx.tolist(), "test_indices": test_idx.tolist(),
            "best_params": dict(search.best_params_),
            "best_inner_score": float(search.best_score_), "outer_score": score,
        })
        print(f"  Inner best: {search.best_score_:.4f}; outer accuracy: {score:.4f}; "
              f"params: {search.best_params_}; elapsed={perf_counter() - started:.1f}s", flush=True)
        # 不让上一折搜索对象/最佳模型在下一折搜索时继续占用内存。
        del search, train_data, train_labels

    average_score = float(np.average(scores))
    if return_details:
        return average_score, scores, details
    return average_score, scores