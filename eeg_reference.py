"""阶段 2/3 共用的电极布局、坏通道修复、平均参考和秩校验。

只操作副本；检测副本可去掉坏样本，但正式 Raw 始终保留完整时间轴。

这是共用功能模块，不是一键运行入口：阶段 2 调用 repair_and_reference 做修复，
阶段 3 验证参考记录，阶段 4 验证平均参考仍成立。
坏段是“某段时间不可靠”；坏通道是“某个电极持续不可靠”，二者处理不能互相替代。
顺序必须是：标准化电极 -> 用好样本检测坏通道 -> 插值 -> 即时平均参考 -> 估计有效秩。
平均参考是在每个时间点，从各通道减去该时间点所有通道的平均值，并非时间基线校正。
"""
from __future__ import annotations

import json
import warnings

import mne
import numpy as np

import config


def _json_default(value):
    """将 NumPy 标量转换为可写入 FIF description 的原生 Python 类型。"""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def prepare_channels(raw):
    """返回 (带标准电极坐标的 Raw 副本, 原名到标准名的映射)，保持通道排列顺序。

    完整历史布局才做设备编号映射；已标准化的 Cz 不二次误映射。
    坐标用于球面样条插值和 ICA 头皮图；未知/混合命名报错，不能猜测电极位置。
    """
    prepared = raw.copy()
    names = prepared.ch_names
    normalized = [name.strip().upper() for name in names]
    montage = mne.channels.make_standard_montage(config.ICA_MONTAGE)
    if len(config.ICA_DEVICE_CHANNELS) != 32 or len(set(config.ICA_DEVICE_CHANNELS)) != 32:
        raise ValueError("ICA_DEVICE_CHANNELS must contain 32 unique names")
    # 此处 CZ 是历史设备的第 1 号标签，不意味着它在标准头顶 Cz 的位置。
    device_names = ["CZ"] + [str(index) for index in range(2, 33)]
    eeg_names = [f"EEG{index}" for index in range(1, 33)]
    if len(names) == 32 and set(normalized) in (set(device_names), set(eeg_names)):
        original = device_names if set(normalized) == set(device_names) else eeg_names
        lookup = dict(zip(original, config.ICA_DEVICE_CHANNELS))
        mapping = {old: lookup[key] for old, key in zip(names, normalized)}
    else:
        if not set(names).issubset(montage.ch_names):
            raise ValueError("Unknown/mixed channel layout; refusing an ambiguous device mapping")
        mapping = {name: name for name in names}
    if set(prepared.get_channel_types()) != {"eeg"}:
        raise ValueError("Expected EEG channels only")
    prepared.rename_channels(mapping)
    prepared.set_montage(montage, on_missing="raise")
    positions = np.array([ch["loc"][:3] for ch in prepared.info["chs"]])
    if not np.isfinite(positions).all() or np.any(np.linalg.norm(positions, axis=1) == 0):
        raise ValueError("Missing/invalid EEG electrode positions")
    if len(np.unique(positions, axis=0)) != len(positions):
        raise ValueError("Duplicate EEG electrode positions")
    return prepared, mapping


def analysis_data(raw, padding):
    """从临时副本提取未被 bad* 覆盖的样本；边界扩展不写回原始注释。"""
    # 返回（通道，剩余样本）的伏单位数组；时间上可能不连续，只用于统计，不能替代正式 Raw。
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError("Boundary padding must be positive")
    view = raw.copy()
    annotations = raw.annotations.copy()
    for index, label in enumerate(annotations.description):
        if label == config.BAD_BOUNDARY_LABEL:
            start = max(0.0, annotations.onset[index] - padding)
            stop = min(raw.n_times / raw.info["sfreq"], annotations.onset[index]
                       + annotations.duration[index] + padding)
            annotations.onset[index] = start
            annotations.duration[index] = stop - start
    view.set_annotations(annotations)
    return view.get_data(picks=list(range(len(raw.ch_names))), reject_by_annotation="omit")


def numerical_rank(data):
    """与 ICA 的中心化 PCA 一致，显式容差避免把插值残差当作独立信号。"""
    tol = config.EEG_RANK_REL_TOL
    if isinstance(tol, bool) or not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("EEG_RANK_REL_TOL must be between 0 and 1")
    if data.shape[1] < 2 or not np.isfinite(data).all():
        raise ValueError("Insufficient or non-finite samples for rank estimation")
    # 每个通道先减去时间均值再做 SVD；数值秩表示独立维度，不等于电极个数。
    # 插值不会创造独立信号，平均参考也会至少减少一个维度。
    singular = np.linalg.svd(data - data.mean(axis=1, keepdims=True), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * tol)) if singular[0] > 0 else 0
    return rank, singular.tolist()


def check_average_reference(raw):
    """只检查、不施加参考：要求已参考、无待应用投影、无未处理坏通道，且逐时刻通道均值近零。"""
    if not raw.info["custom_ref_applied"] or raw.info["projs"] or raw.info["bads"]:
        raise ValueError("Expected applied average reference, no projections and no unresolved bad channels")
    # 标志位本身不能证明数据已做平均参考，因此还要分块检查实际数值（容差单位为伏）。
    for start in range(0, raw.n_times, 100000):
        data = raw.get_data(start=start, stop=min(start + 100000, raw.n_times))
        if not np.isfinite(data).all() or not np.allclose(data.mean(axis=0), 0, atol=1e-12, rtol=0):
            raise ValueError("EEG data violates the average reference constraint")


def validate_reference_record(raw):
    """拒绝旧未参考输入及仅修改标志位的伪平均参考输入。"""
    # 除数值约束，还要求阶段 2 的来源记录与当前通道顺序/样本数匹配，返回该参考记录。
    check_average_reference(raw)
    try:
        metadata = json.loads(raw.info.get("description") or "{}")
        record = metadata["eeg_reference"]
        rank = record["eeg_rank"]
        valid = (metadata["stage"] == 2 and record["schema_version"] == 1
                 and record["reference"] == "average" and record["projection"] is False
                 and record["channel_names"] == raw.ch_names
                 and record["n_times"] == raw.n_times
                 and type(rank) is int and 1 <= rank < len(raw.ch_names))
    except (TypeError, ValueError, KeyError):
        valid = False
    if not valid:
        raise ValueError("Missing/invalid stage 2 average reference record; rerun EEG_preprocessing.py")
    return record


def detect_bad_channels(raw):
    """返回 (最终坏通道名列表, 检测指标字典, 排除坏段后的分析数组)，不修改 raw。

    先查平线和异常标准差，再对剩余候选做 LOF（局部离群因子）比较。
    最终集合 = (自动检测 ∪ 已有标坏 ∪ 人工加入) - 人工撤销，并保持原通道顺序。
    阈值判断是质量筛查而不是确定诊断，过多坏通道或供体不足时停止而非强行插值。
    """
    names = raw.ch_names
    if len(names) < 2:
        raise ValueError("Average reference needs at least two EEG channels")
    for name in ("BAD_CHANNEL_LOF_N_NEIGHBORS", "BAD_CHANNEL_MIN_GOOD_CHANNELS"):
        value = getattr(config, name)
        if type(value) is not int or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    if type(config.BAD_CHANNEL_MAX_COUNT) is not int or config.BAD_CHANNEL_MAX_COUNT < 0:
        raise ValueError("BAD_CHANNEL_MAX_COUNT must be a nonnegative integer")
    for name in ("BAD_CHANNEL_LOF_THRESHOLD", "BAD_CHANNEL_FLAT_STD_UV",
                 "BAD_CHANNEL_MIN_STD_RATIO", "BAD_CHANNEL_MAX_STD_RATIO",
                 "BAD_CHANNEL_MIN_GOOD_SECONDS"):
        value = getattr(config, name)
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.BAD_CHANNEL_LOF_THRESHOLD <= 1 or not (
            config.BAD_CHANNEL_MIN_STD_RATIO < 1 < config.BAD_CHANNEL_MAX_STD_RATIO):
        raise ValueError("Invalid LOF threshold or channel standard deviation ratios")
    overrides = []
    for key in ("BAD_CHANNELS_MANUAL_INCLUDE", "BAD_CHANNELS_MANUAL_EXCLUDE"):
        values = getattr(config, key)
        if not isinstance(values, (tuple, list)) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"{key} must contain standard EEG channel names")
        if len(values) != len(set(values)) or not set(values).issubset(names):
            raise ValueError(f"{key}: duplicate or unknown channel names")
        overrides.append(set(values))
    include, exclude = overrides
    if include & exclude:
        raise ValueError("Manual include and exclude must not overlap")
    data = analysis_data(raw, config.BAD_CHANNEL_BOUNDARY_PADDING_SECONDS)
    if data.shape[1] < max(2, int(np.ceil(config.BAD_CHANNEL_MIN_GOOD_SECONDS * raw.info["sfreq"]))):
        raise ValueError("Not enough unannotated data for bad channel detection; inspect bad annotations")
    if not np.isfinite(data).all():
        raise ValueError("Non-finite data cannot be repaired by automatic interpolation")
    # axis=1 沿时间轴求标准差；使用所有通道标准差的中位数作为相对尺度参照。
    std = data.std(axis=1)
    median = float(np.median(std))
    flat = std <= config.BAD_CHANNEL_FLAT_STD_UV * 1e-6
    low = std < median * config.BAD_CHANNEL_MIN_STD_RATIO
    high = std > median * config.BAD_CHANNEL_MAX_STD_RATIO
    deterministic = [name for i, name in enumerate(names) if flat[i] or low[i] or high[i]]
    # LOF 不自动拒绝 annotations；使用已剔除坏样本的专用 RawArray。
    # 已确定异常/人工加入的通道不再当 LOF 邻居，避免明显坏通道干扰其他通道评分。
    eligible = [i for i, name in enumerate(names) if name not in deterministic
                and name not in raw.info["bads"] and name not in include]
    lof_bads, scores = [], {}
    neighbors = None
    if len(eligible) >= 4:
        neighbors = min(config.BAD_CHANNEL_LOF_N_NEIGHBORS, len(eligible) - 1)
        detection = mne.io.RawArray(data[eligible], mne.create_info(
            [names[i] for i in eligible], raw.info["sfreq"], "eeg"), verbose=False)
        lof_bads, values = mne.preprocessing.find_bad_channels_lof(
            detection, n_neighbors=neighbors, threshold=config.BAD_CHANNEL_LOF_THRESHOLD,
            return_scores=True, verbose=False)
        scores = dict(zip(detection.ch_names, np.abs(values).tolist()))
    else:
        warnings.warn("Fewer than four LOF candidates: skipping LOF; using flat/scale/manual checks", RuntimeWarning)
    # MANUAL_EXCLUDE 是撤销“坏”的判定，不是从信号数组删除电极。
    automatic = set(deterministic) | set(lof_bads)
    final = [name for name in names if name in (automatic | set(raw.info["bads"]) | include) - exclude]
    if len(final) > config.BAD_CHANNEL_MAX_COUNT:
        raise ValueError(f"Too many bad channels ({final}); inspect data and thresholds before interpolation")
    if final and len(names) - len(final) < config.BAD_CHANNEL_MIN_GOOD_CHANNELS:
        raise ValueError("Not enough good donor channels for EEG interpolation")
    metrics = {name: dict(std_uv=float(std[i] * 1e6), flat=bool(flat[i]),
                         low_scale=bool(low[i]), high_scale=bool(high[i]), lof_score=scores.get(name))
               for i, name in enumerate(names)}
    return final, dict(metrics=metrics, automatic=sorted(automatic), existing=list(raw.info["bads"]),
                      manual_include=sorted(include), manual_exclude=sorted(exclude),
                      usable_samples=int(data.shape[1]), lof_neighbors=neighbors,
                      lof_skipped=neighbors is None), data


def repair_and_reference(raw):
    """拼接后调用：返回已插值/平均参考的独立副本及完整质量记录。"""
    if raw.info["custom_ref_applied"] or raw.info["projs"]:
        raise ValueError("Stage 2 reference must be applied exactly once, after concatenation")
    for start in range(0, raw.n_times, 100000):
        if not np.isfinite(raw.get_data(start=start, stop=min(start + 100000, raw.n_times))).all():
            raise ValueError("Non-finite input data")
    prepared, mapping = prepare_channels(raw)
    bads, detection, good = detect_bad_channels(prepared)
    before_rank, _ = numerical_rank(good)
    # 告诉 MNE 哪些电极需要由其他好电极估计；reset_bads=True 在修复后清空未解决坏通道列表。
    prepared.info["bads"] = bads
    if bads:
        prepared.interpolate_bads(reset_bads=True, method=dict(eeg="spline"), verbose=False)
    interpolated = analysis_data(prepared, config.BAD_CHANNEL_BOUNDARY_PADDING_SECONDS)
    interpolated_rank, _ = numerical_rank(interpolated)
    # 必须先修复再平均，避免坏电极污染所有通道；projection=False 表示现在就修改数据。
    prepared.set_eeg_reference("average", projection=False, ch_type="eeg", verbose=False)
    check_average_reference(prepared)
    final_data = analysis_data(prepared, config.BAD_CHANNEL_BOUNDARY_PADDING_SECONDS)
    rank, singular = numerical_rank(final_data)
    if rank < 0 or rank >= len(prepared.ch_names):
        raise ValueError(f"Invalid EEG rank after average reference: {rank}")
    # 同时记录 MNE 默认经验秩；ICA 上限使用更保守的中心化有效秩。
    rank_raw = mne.io.RawArray(final_data - final_data.mean(axis=1, keepdims=True),
                              prepared.info.copy(), verbose=False)
    mne_rank = mne.compute_rank(rank_raw, rank=None, proj=False, tol="auto", verbose=False)["eeg"]
    for key in ("onset", "duration", "description"):
        np.testing.assert_array_equal(getattr(prepared.annotations, key), getattr(raw.annotations, key))
    # 完整报告包括检测分数、插值通道和各阶段秩；调用方只把必要摘要嵌入 FIF description。
    record = dict(schema_version=1, reference="average", projection=False,
                  channel_names=prepared.ch_names, n_times=prepared.n_times,
                  montage=config.ICA_MONTAGE, channel_mapping=mapping,
                  interpolated_channels=bads, interpolation_method="spline",
                  eeg_rank=rank, rank_before=before_rank, rank_after_interpolation=interpolated_rank,
                  mne_rank_auto=int(mne_rank), rank_rel_tol=config.EEG_RANK_REL_TOL,
                  singular_values=singular, rank_method="centered SVD relative tolerance",
                  detection=detection, settings={key: getattr(config, key) for key in dir(config)
                                                if key.startswith("BAD_CHANNEL")},
                  mne_version=mne.__version__)
    print(f"Stage 2 EEG: interpolated={bads}, average reference, effective rank={rank}/{len(prepared.ch_names)}")
    return prepared, record