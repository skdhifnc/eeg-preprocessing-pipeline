# -*- coding: utf-8 -*-
"""阶段 3：fit -> inspect -> 修改 config -> preview -> apply。

拟合、检查、预览和正式发布互相独立。缓存只在输入指纹、参数、模型哈希
均匹配时复用；GUI 候选不写回正式模型，config 是排除编号的唯一来源。
ICA 只能减去选定成分，不能保证所有伪迹消失；重检幅值坏段并保留永久坏段供最终分段。

输入是阶段 2 已插值/平均参考的连续 Raw，不是阶段 1 的 QC Epoch。
fit：学习分解模型并保存模型/清单；review（inspect）：看图挑候选；
preview：按人工配置在副本上重建、画前后对比并记下预览凭据；
apply：验证同一输入、模型、排除列表已经预览，且获得人工确认后发布 clean Raw。
成分数是分解空间的维数，不是重建后保留的 EEG 通道数；本文件不做最终带通和分段。
无命令行参数时按 config.ICA_RUN_MODE 只运行一个步骤，不自动推进或代替人工确认。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import uuid
import warnings

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mne
import numpy as np
from sklearn.exceptions import ConvergenceWarning

import config
from eeg_reference import prepare_channels as _prepare_channels, validate_reference_record
from EEG_preprocessing import amplitude_settings, annotate_bad_amplitude, merge_intervals


# 历史设备编号表，不是把真实标准 Cz 电极映射到 Fp1 的通用规则！
DEVICE_CHANNELS = config.ICA_DEVICE_CHANNELS


def utc_now():
    """生成带时区的 UTC 时间，记录拟合、预览、发布发生的时间。"""
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    """分块计算文件内容 SHA-256 指纹；即使文件名相同，内容变化也能检测到。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, content):
    """独立 JSON 原子替换；模型与 manifest 两文件通过哈希失配拒绝误用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ica-json-", dir=path.parent) as folder:
        temporary = Path(folder) / "record.json"
        temporary.write_text(json.dumps(content, indent=2, ensure_ascii=True), encoding="utf-8")
        os.replace(temporary, path)


def read_json(path):
    """读取模型清单/审核凭据；缺失或 JSON 损坏时不猜测默认内容。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_annotation_roundtrip(original, restored, sfreq):
    """按完整记录校验 FIF 往返注释，允许同 onset 的记录被 MNE 重新排序。

    不能分别按数组位置比较 onset/duration：任务事件和坏段可能同 onset，FIF
    回读后的次序不属于注释语义。这里按名称、通道范围、onset、duration 规范
    排序后配对；文本/通道必须精确一致，时间仍只容许一个采样周期的量化误差。
    """
    if type(sfreq) not in (int, float) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Annotation round-trip sfreq must be positive and finite")

    def canonical(annotations):
        count = len(annotations)
        channel_names = getattr(annotations, "ch_names", [()] * count)
        records = [
            (str(description), tuple(channels), float(onset), float(duration))
            for description, channels, onset, duration in zip(
                annotations.description, channel_names, annotations.onset, annotations.duration
            )
        ]
        # 名称和通道先分组，再按时间排序；同 onset 时按 duration 给出确定次序。
        return sorted(records, key=lambda item: (item[0], item[1], item[2], item[3]))

    expected = canonical(original)
    actual = canonical(restored)
    if [(item[0], item[1]) for item in actual] != [(item[0], item[1]) for item in expected]:
        raise RuntimeError("Clean FIF annotation names/channel scopes changed during round-trip")
    if not expected:
        return
    tolerance = 1.0 / float(sfreq)
    try:
        np.testing.assert_allclose(
            [item[2] for item in actual], [item[2] for item in expected],
            atol=tolerance, rtol=0,
        )
        np.testing.assert_allclose(
            [item[3] for item in actual], [item[3] for item in expected],
            atol=tolerance, rtol=0,
        )
    except AssertionError as exc:
        raise RuntimeError("Clean FIF annotation timing changed during round-trip") from exc


def fit_settings():
    """验证支持的拟合参数，并返回可与 manifest 比较的参数快照（含 MNE 版本）。"""
    if len(config.ICA_DEVICE_CHANNELS) != 32 or len(set(config.ICA_DEVICE_CHANNELS)) != 32:
        raise ValueError("ICA_DEVICE_CHANNELS must contain 32 unique standard channel names")
    if config.ICA_METHOD != "fastica":
        raise ValueError("This workflow currently supports the verified FastICA backend only")
    if config.ICA_DECIM is not None or not config.ICA_REJECT_BY_ANNOTATION:
        raise ValueError("Keep ICA_DECIM=None and ICA_REJECT_BY_ANNOTATION=True")
    padding = config.ICA_BOUNDARY_PADDING_SECONDS
    if not np.isfinite(padding) or padding <= 0:
        raise ValueError("ICA_BOUNDARY_PADDING_SECONDS must be positive")
    if isinstance(config.ICA_N_COMPONENTS, bool) or not isinstance(config.ICA_N_COMPONENTS, int):
        raise ValueError("ICA_N_COMPONENTS must be an integer >= 2")
    if config.ICA_N_COMPONENTS < 2:
        raise ValueError("ICA_N_COMPONENTS must be >= 2")
    return dict(
        method=config.ICA_METHOD, n_components=config.ICA_N_COMPONENTS,
        random_state=config.ICA_RANDOM_STATE, max_iter=config.ICA_MAX_ITER,
        decim=config.ICA_DECIM, reject_by_annotation=True, boundary_padding=padding,
        montage=config.ICA_MONTAGE, device_channels=list(config.ICA_DEVICE_CHANNELS),
        mne_version=mne.__version__,
    )


def prepare_channels(raw):
    """只接受完整历史设备布局或已经标准化的命名，防止 CZ 二次误映射。"""
    return _prepare_channels(raw)


def load_input():
    """返回 (阶段 2 Raw 副本, source 指纹记录)，校验高通、相对时间和平均参考。

    读前/读后各算一次哈希，避免把读取期间变化的文件当成稳定输入。
    这里不会重跑阶段 1/2，也不再次施加参考。
    """
    path = Path(config.ICA_INPUT_RAW_PATH).resolve()
    before = file_hash(path)
    with mne.io.read_raw_fif(path, preload=True, verbose=False) as loaded:
        if len(loaded.filenames) != 1:
            raise ValueError("Only single-file stage 2 FIF inputs are supported")
        raw = loaded.copy()
    if before != file_hash(path):
        raise RuntimeError("Stage 2 input changed during loading; retry after preprocessing finishes")
    if raw.first_samp != 0 or raw.info["meas_date"] is not None or raw.annotations.orig_time is not None:
        raise ValueError("Expected stage 2 relative-time Raw with first_samp=0")
    if raw.info["highpass"] < 1.0:
        raise ValueError("Expected >=1 Hz highpass")
    if raw.n_times < 2 or not raw.ch_names:
        raise ValueError("Empty or too short recording")
    for start in range(0, raw.n_times, 100000):
        data = raw.get_data(start=start, stop=min(start + 100000, raw.n_times))
        if not np.isrealobj(data) or not np.isfinite(data).all():
            raise ValueError("Non-finite or complex EEG data")
    reference = validate_reference_record(raw)
    source = dict(path=str(path), sha256=before, n_times=int(raw.n_times),
                  sfreq=float(raw.info["sfreq"]), ch_names=raw.ch_names,
                  eeg_rank=int(reference["eeg_rank"]))
    return raw, source


def make_fit_raw(raw):
    """扩大拟合副本中的接缝窗口；原 Raw 的零时长边界及所有任务事件不变。"""
    # 零时长接缝本身不排除足够的拟合样本；扩展前后窗口后，ICA 才会忽略附近突变。
    fit_settings()
    fit_raw = raw.copy()
    annotations = raw.annotations.copy()
    for index, label in enumerate(annotations.description):
        if label == "bad_boundary":
            start = max(0.0, annotations.onset[index] - config.ICA_BOUNDARY_PADDING_SECONDS)
            stop = min(raw.n_times / raw.info["sfreq"], annotations.onset[index]
                       + annotations.duration[index] + config.ICA_BOUNDARY_PADDING_SECONDS)
            annotations.onset[index] = start
            annotations.duration[index] = stop - start
    fit_raw.set_annotations(annotations)
    return fit_raw


def assert_current(source):
    """发布/审核结束前复查阶段 2 输入哈希，避免处理过程中上游被重新生成。"""
    if file_hash(config.ICA_INPUT_RAW_PATH) != source["sha256"]:
        raise RuntimeError("Stage 2 input changed. Run fit and review again")


def load_matching():
    """只有输入、拟合参数、模型哈希和结构一致时，才返回 (Raw, ICA, manifest)。"""
    raw, source = load_input()
    manifest = read_json(config.ICA_MANIFEST_PATH)
    if manifest["source"] != source or manifest["settings"] != fit_settings():
        raise RuntimeError("ICA input/settings are stale. Run fit and review again")
    if file_hash(config.ICA_MODEL_PATH) != manifest["model_sha256"]:
        raise RuntimeError("ICA model/manifest mismatch. Run fit --force")
    ica = mne.preprocessing.read_ica(config.ICA_MODEL_PATH, verbose=False)
    expected = [name for name in raw.ch_names if name not in raw.info["bads"]]
    if ica.ch_names != expected or ica.n_components_ != manifest["n_components"]:
        raise RuntimeError("ICA channel/component mismatch")
    ica.exclude = []  # 不信任缓存或 GUI 的隐式排除状态
    return raw, ica, manifest


def report_dir(manifest):
    """创建并返回当前模型 ID 专属报告目录，避免新旧模型的审核记录混用。"""
    folder = Path(config.ICA_REPORT_DIR) / manifest["model_id"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def overview(raw, ica, folder, show=False):
    """每页最多 15 个成分；浏览用副本，点击不影响正式模型。"""
    view = ica.copy()
    for start in range(0, ica.n_components_, 15):
        picks = list(range(start, min(start + 15, ica.n_components_)))
        fig = view.plot_components(picks=picks, inst=raw, ncols=min(5, len(picks)),
                                   title="ICA components (zero-based indices)", show=False)
        fig.savefig(folder / f"ica_components_{start:03d}.png", dpi=150)
        if show:
            plt.show(block=True)
        plt.close(fig)


def fit_stage(force=False):
    """拟合或复用模型，返回 (ICA, manifest)；此步骤不减去成分、不输出 clean EEG。"""
    settings = fit_settings()
    if not force and Path(config.ICA_MODEL_PATH).exists() and Path(config.ICA_MANIFEST_PATH).exists():
        try:
            raw, ica, manifest = load_matching()
        except RuntimeError:
            print("Input or settings changed; fitting a new ICA model.")
        else:
            overview(raw, ica, report_dir(manifest))
            print(f"Reusing matching model {manifest['model_id']}; --force refits it")
            return ica, manifest
    raw, source = load_input()
    fit_raw = make_fit_raw(raw)
    picks = mne.pick_types(fit_raw.info, eeg=True, exclude="bads")
    good = fit_raw.get_data(picks=picks, reject_by_annotation="omit")
    if good.shape[1] < 2 or len(picks) < config.ICA_N_COMPONENTS:
        raise ValueError("Not enough good channels or unannotated samples for ICA")
    # 再对实际拟合用的好样本检查数值秩，避免用超过独立信息量的成分数拟合。
    rank = np.linalg.matrix_rank(good - good.mean(axis=1, keepdims=True))
    if config.ICA_N_COMPONENTS > rank:
        raise ValueError(f"ICA_N_COMPONENTS exceeds the numerical rank ({rank})")
    usable = good.shape[1]
    del good
    ica = mne.preprocessing.ICA(n_components=config.ICA_N_COMPONENTS, method=config.ICA_METHOD,
                               random_state=config.ICA_RANDOM_STATE, max_iter=config.ICA_MAX_ITER)
    # 未收敛警告升级为异常：不将未收敛模型发布为可审核的正式模型。
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        ica.fit(fit_raw, picks=picks, reject_by_annotation=True, decim=None)
    if ica.n_samples_ != usable:
        raise RuntimeError("ICA did not fit exactly the unannotated samples")
    # 新拟合生成新 model_id；旧模型成分编号不可直接当成新模型的相同伪迹。
    manifest = dict(model_id=uuid.uuid4().hex, fitted_at_utc=utc_now(), source=source,
                    settings=settings, n_components=int(ica.n_components_),
                    n_samples=int(ica.n_samples_), n_iter=int(ica.n_iter_), converged=True,
                    omitted_seconds=(raw.n_times - usable) / raw.info["sfreq"])
    model_path = Path(config.ICA_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="ica-fit-", dir=model_path.parent) as folder:
        temporary = Path(folder) / "model-ica.fif"
        ica.save(temporary, overwrite=False, verbose=False)
        restored = mne.preprocessing.read_ica(temporary, verbose=False)
        if restored.ch_names != ica.ch_names or restored.n_components_ != ica.n_components_:
            raise RuntimeError("Saved ICA model failed validation")
        manifest["model_sha256"] = file_hash(temporary)
        assert_current(source)
        # 模型和清单各自原子替换，但不是两文件事务；中断失配会在 load_matching 中被拒绝。
        os.replace(temporary, model_path)
        write_json(config.ICA_MANIFEST_PATH, manifest)
    overview(raw, ica, report_dir(manifest))
    print(f"Model {manifest['model_id']}: components={ica.n_components_}, "
          f"samples={usable}/{raw.n_times}, omitted={manifest['omitted_seconds']:.3f}s, "
          f"iterations={ica.n_iter_}, converged=True")
    return ica, manifest


def validate_exclusions(values, n_components):
    """验证 0-based 成分编号，拒绝重复、越界和全部排除，返回排序后的 Python 整数列表。"""
    values = list(values)
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError("ICA indices must be integers (ICA000 = 0)")
    if len(values) != len(set(values)) or any(value < 0 or value >= n_components for value in values):
        raise ValueError(f"Duplicate or out-of-range ICA indices; valid range: 0..{n_components - 1}")
    if len(values) == n_components:
        raise ValueError("Refusing to exclude every ICA component")
    return sorted(int(value) for value in values)


def inspect_stage(components=None, show=True, prompt=True):
    """审核头皮图、源时序和候选属性图，返回候选编号，不改变正式排除配置。

    components 是要看详细属性的编号；show=False 只导出静态图。
    prompt 只控制命令行交互询问，VS Code 配置入口关闭它，避免按钮运行等待终端输入。
    """
    raw, ica, manifest = load_matching()
    folder = report_dir(manifest)
    overview(raw, ica, folder, show=show)
    # GUI 点选只操作这个副本；最终要排除谁仍需人工填写 config.ICA_EXCLUDE_COMPONENTS。
    view = ica.copy()
    view.exclude = validate_exclusions(config.ICA_EXCLUDE_COMPONENTS, ica.n_components_)
    if show:
        # 强制使用已有 matplotlib 浏览器，避免依赖未安装的 mne-qt-browser。
        page_size = int(config.ICA_SOURCE_VISIBLE_COMPONENTS)
        if page_size < 1 or config.ICA_SOURCE_WINDOW_SECONDS <= 0:
            raise ValueError("Source display size/window must be positive")
        with mne.viz.use_browser_backend("matplotlib"):
            for start in range(0, ica.n_components_, page_size):
                view.plot_sources(raw, picks=list(range(start, min(start + page_size, ica.n_components_))),
                                  start=0.0, stop=min(config.ICA_SOURCE_WINDOW_SECONDS, raw.times[-1]),
                                  title="ICA candidates: click trace; close window to continue",
                                  block=True)
    candidates = validate_exclusions(view.exclude, ica.n_components_)
    write_json(folder / "candidate_components.json", dict(model_id=manifest["model_id"],
               source_sha256=manifest["source"]["sha256"], candidates=candidates, created_at=utc_now()))
    print(f"Candidate indices (not applied): {candidates}")
    if components is None and show and prompt:
        answer = input("Detail indices, comma-separated; Enter uses candidates: ").strip()
        components = [int(part.strip()) for part in answer.split(",")] if answer else candidates
    details = validate_exclusions(components if components is not None else candidates, ica.n_components_)
    # 属性图的分段只用于诊断成分，不是按 left/right 切的运动想象试次。
    fit_raw = make_fit_raw(raw)
    upper = min(config.ICA_PSD_FMAX, raw.info["sfreq"] / 2)
    if not 0 <= config.ICA_PSD_FMIN < upper:
        raise ValueError("Invalid ICA PSD limits")
    for component in details:
        figures = view.plot_properties(fit_raw, picks=[component], reject=None,
                                      reject_by_annotation=True, figsize=(11, 8), show=False,
                                      psd_args=dict(fmin=config.ICA_PSD_FMIN, fmax=upper))
        for figure in figures:
            # MNE 属性图使用显式 axes 位置，subplots_adjust 不会为标题留白。
            for axis in figure.axes:
                position = axis.get_position()
                axis.set_position([position.x0, position.y0 * 0.94,
                                   position.width, position.height * 0.94])
            figure.suptitle(f"ICA{component:03d} properties (2-second segments, not task ERP)",
                           y=0.99, fontsize=13)
            figure.savefig(folder / f"ICA{component:03d}_properties.png", dpi=150)
            if show:
                plt.show(block=True)
            plt.close(figure)
    assert_current(manifest["source"])
    print("Write final zero-based indices into config.ICA_EXCLUDE_COMPONENTS, then run preview.")
    return candidates


def reconstruct(raw, ica, excludes):
    """在模型和 Raw 的副本上减去选定成分，返回保留全部通道/样本/注释的 clean Raw。"""
    selected = validate_exclusions(excludes, ica.n_components_)
    if not selected:
        warnings.warn("No ICA components selected: this is NOT an artifact-removal decision", RuntimeWarning)
    model = ica.copy()
    model.exclude = selected
    # apply 不把 EEG 降成 ICA_N_COMPONENTS 个通道；按原传感器空间重建并保留 PCA 残余。
    clean = model.apply(raw.copy(), verbose=False)
    if clean.n_times != raw.n_times or clean.ch_names != raw.ch_names:
        raise RuntimeError("ICA reconstruction changed the time axis or channels")
    for key in ("onset", "duration", "description"):
        np.testing.assert_array_equal(getattr(clean.annotations, key), getattr(raw.annotations, key))
    if not np.isfinite(clean.get_data()).all():
        raise RuntimeError("Non-finite reconstruction")
    return clean


def recheck_amplitude(clean):
    """原地替换可复检的幅值标记；不改样本、任务或其他永久质量标记。"""
    settings = amplitude_settings(post_ica=True)
    replaceable = {config.PREICA_BAD_AMPLITUDE_LABEL, "bad_amplitude"}
    indices = [i for i, name in enumerate(clean.annotations.description) if name in replaceable]

    def summary(annotations):
        intervals = merge_intervals((a["onset"], a["onset"] + a["duration"]) for a in annotations)
        return dict(count=len(annotations), duration_seconds=sum(end - start for start, end in intervals),
                    annotations=[dict(onset=float(a["onset"]), duration=float(a["duration"]),
                                      description=str(a["description"])) for a in annotations])

    previous = summary(clean.annotations[indices])
    # 在副本上完成检测再提交；异常时不让调用者持有已删除初筛注释的 Raw。
    candidate = clean.copy()
    candidate.annotations.delete(indices)
    permanent = summary(candidate.annotations[[i for i, name in enumerate(candidate.annotations.description)
                                               if name.lower().startswith(("bad", "edge"))]])
    detected = annotate_bad_amplitude(candidate, settings=settings)
    clean.set_annotations(candidate.annotations)
    record = dict(schema_version=1, settings=settings, pre_ica=previous,
                  post_ica=summary(detected), permanent=permanent)
    print(f"Amplitude recheck: pre={previous['count']} ({previous['duration_seconds']:.3f}s), "
          f"post={len(detected)} ({record['post_ica']['duration_seconds']:.3f}s), "
          f"permanent={permanent['count']}")
    return record


def comparison_figure(raw, clean):
    """共同滑块驱动两个轴；同通道、同时间、同纵轴比例，标记任务和坏段。"""
    names = [name for name in config.ICA_PREVIEW_CHANNELS if name in raw.ch_names]
    if not names:
        names = raw.ch_names[:5]
    before, after = raw.get_data(picks=names) * 1e6, clean.get_data(picks=names) * 1e6
    spacing = max(float(np.percentile(np.abs(before), 99)) * 3, 20.0)
    offsets = np.arange(len(names)) * spacing
    duration = min(config.ICA_SOURCE_WINDOW_SECONDS, raw.times[-1])
    if duration <= 0:
        raise ValueError("Preview window must be positive")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.15, hspace=0.25)

    def draw(start):
        first = int(start * raw.info["sfreq"])
        last = min(raw.n_times, first + int(duration * raw.info["sfreq"]) + 1)
        for ax, data, title, annotated in zip(axes, (before, after), ("Before ICA", "After ICA"), (raw, clean)):
            ax.clear()
            for index in range(len(names)):
                ax.plot(raw.times[first:last], data[index, first:last] + offsets[index], lw=0.6)
            for annotation in annotated.annotations:
                onset, end = annotation["onset"], annotation["onset"] + annotation["duration"]
                if end < start or onset > start + duration:
                    continue
                if annotation["description"].lower().startswith("bad"):
                    if end > onset:
                        ax.axvspan(onset, end, color="red", alpha=0.15)
                    else:
                        ax.axvline(onset, color="red", linestyle="--", alpha=0.5)
                else:
                    ax.axvline(onset, color="gray", alpha=0.3)
                    ax.text(onset, offsets[-1] + spacing * 0.6, annotation["description"], fontsize=8)
            ax.set_yticks(offsets, names)
            ax.set_ylim(-spacing, offsets[-1] + spacing)
            ax.set_xlim(start, start + duration)
            ax.set_ylabel(f"EEG (uV), offset={spacing:.1f}")
            ax.set_title(title)
        axes[-1].set_xlabel("Recording time (s)")
        fig.canvas.draw_idle()

    maximum = raw.times[-1] - duration
    if maximum > 0:
        slider = Slider(fig.add_axes([0.16, 0.04, 0.7, 0.03]), "Start (s)", 0, maximum, valinit=0)
        slider.on_changed(draw)
        fig._time_slider = slider  # 保留交互组件引用
    draw(0.0)
    return fig


def selection_record(manifest, selected):
    """绑定模型 ID、模型/输入哈希和排除列表，作为 preview 与 apply 必须一致的凭据字段。"""
    return dict(model_id=manifest["model_id"], model_sha256=manifest["model_sha256"],
                input_sha256=manifest["source"]["sha256"], excluded_components=selected,
                postica_amplitude_settings=amplitude_settings(post_ica=True))


def preview_stage(show=True):
    """返回预览用 clean 副本并保存对比图/凭据；不写正式 clean FIF，也不自动确认效果。"""
    raw, ica, manifest = load_matching()
    selected = validate_exclusions(config.ICA_EXCLUDE_COMPONENTS, ica.n_components_)
    clean = reconstruct(raw, ica, selected)
    amplitude_recheck = recheck_amplitude(clean)
    folder = report_dir(manifest)
    receipt = folder / "preview.json"
    # 先让旧凭据失效；本次预览未完整结束时，不能凭上次记录发布新的选择。
    receipt.unlink(missing_ok=True)
    overlay = ica.plot_overlay(raw, exclude=selected, start=0.0,
                               stop=min(config.ICA_SOURCE_WINDOW_SECONDS, raw.times[-1]), show=False)
    overlay.savefig(folder / "ica_overlay.png", dpi=150)
    if show:
        plt.show(block=True)
    plt.close(overlay)
    comparison = comparison_figure(raw, clean)
    comparison.savefig(folder / "before_after.png", dpi=150)
    if show:
        plt.show(block=True)
    plt.close(comparison)
    assert_current(manifest["source"])
    record = selection_record(manifest, selected)
    record.update(previewed_at=utc_now(), interactive=show, amplitude_recheck=amplitude_recheck)
    write_json(receipt, record)
    print(f"Preview only; excluded={selected}. Inspect plots before apply --confirm.")
    return clean


def save_clean(raw, manifest, selected, amplitude_recheck):
    """同磁盘临时 FIF 完整回读后发布；模型、输入在发布前再次校验。"""
    output = Path(config.ICA_CLEAN_RAW_PATH).resolve()
    protected = [config.ICA_INPUT_RAW_PATH, config.ICA_MODEL_PATH, config.ICA_MANIFEST_PATH]
    if output in [Path(path).resolve() for path in protected] or not output.name.endswith("_raw.fif"):
        raise ValueError("Clean output must be a separate _raw.fif, never the stage 2 input/model")
    if output.exists() and not config.ICA_CLEAN_OVERWRITE:
        raise FileExistsError(f"Clean output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    record = selection_record(manifest, selected)
    if amplitude_recheck.get("settings") != record["postica_amplitude_settings"]:
        raise ValueError("Amplitude recheck settings changed; preview/apply again")
    if config.PREICA_BAD_AMPLITUDE_LABEL in raw.annotations.description:
        raise ValueError("Pre-ICA amplitude annotations remain; recheck before publication")
    record["amplitude_recheck"] = amplitude_recheck
    record.update(applied_at=utc_now(), input_path=str(Path(config.ICA_INPUT_RAW_PATH).resolve()))
    # 将来源绑定记录嵌入即将保存的 clean；阶段 4 会与外部 applied_components.json 对照。
    raw.info["description"] = json.dumps(record, ensure_ascii=True)
    with TemporaryDirectory(prefix="ica-clean-", dir=output.parent) as folder:
        temporary = Path(folder) / "clean_raw.fif"
        raw.save(temporary, fmt="double", overwrite=False, verbose=False)
        if len(list(Path(folder).glob("*.fif"))) != 1:
            raise RuntimeError("Split FIF output unsupported; previous output retained")
        with mne.io.read_raw_fif(temporary, preload=False, verbose=False) as restored:
            if (restored.n_times != raw.n_times or restored.ch_names != raw.ch_names
                    or restored.info["sfreq"] != raw.info["sfreq"]):
                raise RuntimeError("Clean FIF metadata verification failed")
            validate_annotation_roundtrip(raw.annotations, restored.annotations, raw.info["sfreq"])
            for start in range(0, raw.n_times, 100000):
                stop = min(start + 100000, raw.n_times)
                np.testing.assert_allclose(restored.get_data(start=start, stop=stop),
                                           raw.get_data(start=start, stop=stop), rtol=1e-10, atol=1e-14)
        assert_current(manifest["source"])
        if file_hash(config.ICA_MODEL_PATH) != manifest["model_sha256"]:
            raise RuntimeError("Model changed during apply")
        if output.exists() and not config.ICA_CLEAN_OVERWRITE:
            raise FileExistsError(f"Clean output already exists: {output}")
        os.replace(temporary, output)
    # FIF 内容哈希只能在保存后计算，故仅写外部凭据；两者失配时阶段 4 会拒绝使用。
    record["output_sha256"] = file_hash(output)
    write_json(report_dir(manifest) / "applied_components.json", record)
    print(f"Published ICA reconstruction: {output}")
    return output


def apply_stage(confirm=False):
    """人工确认且存在匹配预览后，重建、正式发布，并返回 clean Raw；不自动选择伪迹。"""
    if not confirm:
        raise ValueError("Review preview first, then set ICA_APPLY_CONFIRM=True "
                         "or use apply --confirm to publish")
    raw, ica, manifest = load_matching()
    selected = validate_exclusions(config.ICA_EXCLUDE_COMPONENTS, ica.n_components_)
    receipt_path = report_dir(manifest) / "preview.json"
    if not receipt_path.exists():
        raise RuntimeError("Run preview for this model and selection before apply")
    receipt = read_json(receipt_path)
    # 确认开关不绕过一致性检查；输入/模型/排除成分任一变化都必须重新预览。
    if any(receipt.get(key) != value for key, value in selection_record(manifest, selected).items()):
        raise RuntimeError("Selection, model or amplitude settings changed since preview; run preview again")
    clean = reconstruct(raw, ica, selected)
    amplitude_recheck = recheck_amplitude(clean)
    save_clean(clean, manifest, selected, amplitude_recheck)
    return clean


def configured_arguments():
    """无命令行参数时读取 VS Code 运行配置，不自动串联阶段或确认发布。"""
    command = config.ICA_RUN_MODE
    if not isinstance(command, str) or command not in {"fit", "review", "inspect", "preview", "apply"}:
        raise ValueError("ICA_RUN_MODE must be fit, review, inspect, preview or apply")
    for name in ("ICA_FORCE_REFIT", "ICA_SHOW_PLOTS", "ICA_APPLY_CONFIRM"):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be True or False (not a string)")
    print(f"ICA config mode: {command}")
    return argparse.Namespace(
        command=command, force=config.ICA_FORCE_REFIT,
        components=config.ICA_REVIEW_COMPONENTS, no_show=not config.ICA_SHOW_PLOTS,
        confirm=config.ICA_APPLY_CONFIRM, prompt=False,
    )


def main(argv=None):
    """无参数（VS Code 运行按钮）使用 config；显式参数保持原命令行行为。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="Fit or reuse matching ICA; does not clean EEG")
    fit.add_argument("--force", action="store_true", help="Refit and invalidate previous reviews")
    inspect = sub.add_parser("inspect", aliases=["review"], help="Maps, source browser, then detailed properties")
    inspect.add_argument("--components", type=int, nargs="*", default=None)
    inspect.add_argument("--no-show", action="store_true", help="Export static plots without GUI/input")
    preview = sub.add_parser("preview", help="Reconstruct for comparison only; no clean FIF output")
    preview.add_argument("--no-show", action="store_true", help="Export plots for manual review later")
    apply = sub.add_parser("apply", help="Publish after a matching preview and human confirmation")
    apply.add_argument("--confirm", action="store_true", help="Confirm you reviewed the selected components")
    # 显式命令行优先；否则使用 VS Code 配置入口。argv 参数方便测试，不必修改系统参数。
    arguments = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(arguments) if arguments else configured_arguments()
    if getattr(args, "no_show", False) or args.command in {"fit", "apply"}:
        # 无 GUI 的步骤使用静态绘图后端，避免 fit/apply 意外弹窗阻塞。
        plt.switch_backend("Agg")
    if args.command == "fit":
        return fit_stage(args.force)
    if args.command in {"inspect", "review"}:
        return inspect_stage(args.components, not args.no_show, prompt=getattr(args, "prompt", True))
    if args.command == "preview":
        return preview_stage(not args.no_show)
    return apply_stage(args.confirm)


if __name__ == "__main__":
    main()