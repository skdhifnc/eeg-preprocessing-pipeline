"""阶段 4 合成数据回归测试；不读取或改写正式数据，不代替人工 ICA 审核。"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use("Agg")
import h5py
import mne
import numpy as np

import config
import EEG_processing as stage4
import ICA_processing as stage3
from ws import load_clean_data


class EpochProcessingTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        settings = dict(
            ICA_INPUT_RAW_PATH=self.root / "stage2_raw.fif",
            ICA_MODEL_PATH=self.root / "model-ica.fif",
            ICA_MANIFEST_PATH=self.root / "manifest.json",
            ICA_CLEAN_RAW_PATH=self.root / "stage3_clean_raw.fif",
            ICA_REPORT_DIR=self.root / "review", ICA_EXCLUDE_COMPONENTS=(2, 0),
            ICA_CLEAN_OVERWRITE=True,
            STAGE4_INPUT_RAW_PATH=self.root / "stage3_clean_raw.fif",
            STAGE4_FULL_OUTPUT_PATH=self.root / "full.hdf5",
            STAGE4_WS_OUTPUT_PATH=self.root / "ws.hdf5",
            STAGE4_REPORT_PATH=self.root / "report.json", STAGE4_OVERWRITE=True,
            STAGE4_EPOCH_TMIN=-1.0, STAGE4_EPOCH_TMAX=4.0,
            STAGE4_WS_TMIN=0.0, STAGE4_WS_TMAX=3.996, STAGE4_BASELINE=None,
            POSTICA_AMPLITUDE_PEAK_UV=100.0, POSTICA_AMPLITUDE_PEAK_MIN_DURATION=0.004,
            POSTICA_ABSOLUTE_AMPLITUDE_THRESHOLD_UV=300.0,
            POSTICA_ABSOLUTE_AMPLITUDE_MIN_DURATION=0.004, POSTICA_AMPLITUDE_MERGE_GAP_SECONDS=0.0,
        )
        for key, value in settings.items():
            context = patch.object(config, key, value)
            context.start()
            self.addCleanup(context.stop)
        # 逆序标准通道验证导出不会偷偷排序。合成 EEG 已做即时平均参考。
        names = list(reversed(config.ICA_DEVICE_CHANNELS))
        data = np.random.default_rng(40).normal(0, 1e-5, (32, 16000))
        self.raw = mne.io.RawArray(data, mne.create_info(names, 250, "eeg"), verbose=False)
        self.raw.set_montage(config.ICA_MONTAGE)
        self.raw.filter(1, None, method="iir", verbose=False)
        self.raw.set_eeg_reference("average", projection=False, verbose=False)
        self.raw.set_annotations(mne.Annotations(
            [.2, 5, 13, 20.5, 21, 29, 30, 37, 45, 53, 61],
            [0, 0, 0, .1, 0, 0, 0, 0, 0, 0, 0],
            ["left", "right", "left", "bad_amplitude", "left", "right",
             "bad_boundary", "left", "right", "left", "right"],
        ))
        self.raw.save(config.ICA_INPUT_RAW_PATH, fmt="double", verbose=False)
        # 阶段 4 不解析 ICA 模型内容，只校验哈希；使用测试字节而非拟合模型。
        Path(config.ICA_MODEL_PATH).write_bytes(b"synthetic model fingerprint fixture")
        self.manifest = dict(model_id="a" * 32, n_components=3, model_sha256=stage3.file_hash(config.ICA_MODEL_PATH),
                             source=dict(sha256=stage3.file_hash(config.ICA_INPUT_RAW_PATH),
                                         ch_names=names, n_times=16000, sfreq=250.0))
        stage3.write_json(config.ICA_MANIFEST_PATH, self.manifest)
        # 使用真实阶段 3 发布函数生成 FIF description 和 apply 凭据。
        clean = self.raw.copy()
        # 复检须依据真实电压，而不是沿用手工标记；成对注入以维持平均参考。
        clean._data[0, 5125:5150] = 500e-6
        clean._data[1, 5125:5150] = -500e-6 - clean._data[2:, 5125:5150].sum(axis=0)
        audit = stage3.recheck_amplitude(clean)
        stage3.save_clean(clean, self.manifest, [0, 2], audit)

    def run_stage4(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return stage4.main()

    def output_bytes(self):
        return {path: path.read_bytes() for path in stage4.output_paths()}

    def test_integration_shapes_units_labels_order_and_source_unchanged(self):
        source_hash = stage3.file_hash(config.ICA_CLEAN_RAW_PATH)
        full, ws = self.run_stage4()
        self.assertEqual(full.get_data().shape, (5, 32, 1251))
        self.assertEqual(ws.get_data().shape, (5, 32, 1000))
        self.assertIsNone(full.baseline)
        np.testing.assert_array_equal(full.events[:, 2], [2, 1, 1, 2, 1])
        np.testing.assert_array_equal(full.selection, [1, 2, 5, 6, 7])
        np.testing.assert_array_equal(ws.get_data(), full.get_data()[:, :, 250:1250])
        self.assertEqual(full.ch_names, self.raw.ch_names)
        with h5py.File(config.STAGE4_FULL_OUTPUT_PATH, "r") as f, h5py.File(
                config.STAGE4_WS_OUTPUT_PATH, "r") as w:
            self.assertEqual(f["clean_eeg"].dtype, np.dtype("float32"))
            self.assertEqual(w["labels"].dtype, np.dtype("int32"))
            self.assertEqual(w["labels"].shape, (5,))
            self.assertEqual(w.attrs["unit"], "microvolts")
            self.assertEqual(w.attrs["sampling_rate"], 250.0)
            self.assertEqual(w.attrs["tmax"], 3.996)
            np.testing.assert_array_equal(f["clean_eeg"][:], (full.get_data() * 1e6).astype("float32"))
            np.testing.assert_array_equal(w["clean_eeg"][:], f["clean_eeg"][:, :, 250:1250])
            for key in ("channel_names", "labels", "event_samples"):
                np.testing.assert_array_equal(f[key][:], w[key][:])
            for key in ("run_id", "source_raw_sha256"):
                self.assertEqual(f.attrs[key], w.attrs[key])
            self.assertEqual(f["channel_names"].asstr()[:].tolist(), self.raw.ch_names)
        data, labels = load_clean_data(config.STAGE4_WS_OUTPUT_PATH)
        self.assertEqual(data.shape, (5, 32, 1000))
        np.testing.assert_array_equal(labels, [2, 1, 1, 2, 1])
        report = stage4.read_json(config.STAGE4_REPORT_PATH)
        self.assertEqual(report["rejection"]["rejected_epochs"], 4)
        self.assertEqual(report["rejection"]["original_labels"], {"1": 5, "2": 4})
        self.assertEqual(report["max_stratified_k_fold"], 2)
        self.assertEqual(report["full_sha256"], stage3.file_hash(config.STAGE4_FULL_OUTPUT_PATH))
        self.assertEqual(report["ws_sha256"], stage3.file_hash(config.STAGE4_WS_OUTPUT_PATH))
        self.assertEqual(stage3.file_hash(config.ICA_CLEAN_RAW_PATH), source_hash)

    def test_terminal_prints_epoch_rejection_summary(self):
        stream = StringIO()
        with redirect_stdout(stream):
            self.run_stage4()
        output = stream.getvalue()
        self.assertIn("候选 Epoch 总数: 9", output)
        self.assertIn("丢弃 Epoch 总数: 4", output)
        self.assertIn("bad*/edge 坏段注释丢弃（按 Epoch 去重）: 2", output)
        self.assertIn("最终保留 Epoch 数: 5", output)
        self.assertIn("- bad_amplitude: 1", output)
        self.assertIn("- bad_boundary: 1", output)
        self.assertIn("Epoch 数: 5（仅裁剪时间，不额外丢弃）", output)

    def test_configurable_windows_labels_unit_and_dtype(self):
        settings = {
            "STAGE4_EVENT_ID": {"left": 7, "right": 9},
            "STAGE4_EPOCH_TMIN": -0.5, "STAGE4_EPOCH_TMAX": 2.0,
            "STAGE4_WS_TMIN": 0.25, "STAGE4_WS_TMAX": 1.25,
            "STAGE4_BASELINE": (-0.25, 0.0),
            "STAGE4_OUTPUT_UNIT": "volts",
            "STAGE4_OUTPUT_DTYPE": "float64",
        }
        patches = [patch.object(config, key, value) for key, value in settings.items()]
        for context in patches:
            context.start()
        try:
            full, ws = self.run_stage4()
            self.assertEqual(full.get_data().shape[2], 626)
            self.assertEqual(ws.get_data().shape[2], 251)
            self.assertEqual(full.baseline, (-0.25, 0.0))
            self.assertEqual(set(full.events[:, 2]), {7, 9})
            with h5py.File(config.STAGE4_WS_OUTPUT_PATH, "r") as h5:
                self.assertEqual(h5["clean_eeg"].dtype, np.dtype("float64"))
                self.assertEqual(h5.attrs["unit"], "volts")
                self.assertEqual(h5.attrs["sampling_rate"], 250.0)
                np.testing.assert_allclose(h5["clean_eeg"][:], ws.get_data())
        finally:
            for context in reversed(patches):
                context.stop()

    def test_rejection_reasons_and_pre_event_bad_segment(self):
        full, _ = self.run_stage4()
        self.assertIn("NO_DATA", full.drop_log[0])
        self.assertIn("bad_amplitude", full.drop_log[3])  # 仅基线区间受污染也拒绝
        self.assertIn("bad_boundary", full.drop_log[4])  # 零时长接缝
        self.assertIn("TOO_SHORT", full.drop_log[8])
        report = stage4.read_json(config.STAGE4_REPORT_PATH)["rejection"]
        self.assertEqual([event["event_index"] for event in report["rejected_events"]], [0, 3, 4, 8])

    def test_filter_frequency_response_and_copy(self):
        raw = self.raw.copy()
        times = raw.times
        signal = sum(np.sin(2 * np.pi * frequency * times) for frequency in (2, 15, 60)) * 1e-5
        raw._data[:] = 0
        raw._data[0] = signal
        raw._data[1] = -signal
        raw.set_annotations(mne.Annotations([], [], []))
        before = raw.get_data().copy()
        filtered = stage4.filter_raw(raw)
        np.testing.assert_array_equal(raw.get_data(), before)
        data = filtered.get_data()[0, 2500:12500]
        t = times[2500:12500]
        amplitudes = [abs(np.dot(data, np.exp(-2j * np.pi * f * t))) * 2 / len(t)
                      for f in (2, 15, 60)]
        self.assertGreater(amplitudes[1], 9e-6)
        self.assertLess(amplitudes[0], 2e-7)
        self.assertLess(amplitudes[2], 2e-7)

    def test_filter_does_not_cross_boundary_or_bad_segment(self):
        raw = self.raw.copy()
        raw._data[:, :7500] = 0
        raw.set_annotations(mne.Annotations([30, 40], [0, 1], ["bad_boundary", "bad_amplitude"]))
        before = raw.get_data().copy()
        filtered = stage4.filter_raw(raw)
        np.testing.assert_array_equal(filtered.get_data()[:, :7500], 0)
        np.testing.assert_array_equal(filtered.get_data()[:, 10000:10250], before[:, 10000:10250])
        for key in ("onset", "duration", "description"):
            np.testing.assert_array_equal(getattr(filtered.annotations, key), getattr(raw.annotations, key))

    def test_missing_input_has_no_fallback(self):
        Path(config.ICA_CLEAN_RAW_PATH).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "run ICA apply first"):
            stage4.main()
        self.assertFalse(any(p.exists() for p in stage4.output_paths()))

    def test_stale_source_model_receipt_or_selection_rejected(self):
        _, source = stage4.load_clean_raw()
        for dependency in source["dependencies"]:
            path = Path(dependency)
            before = path.read_bytes()
            try:
                path.write_bytes(before + b"changed")
                with self.assertRaises(RuntimeError):
                    stage4.verify_source(source)
            finally:
                path.write_bytes(before)
        with patch.object(config, "ICA_EXCLUDE_COMPONENTS", (0,)):
            with self.assertRaisesRegex(ValueError, "selection changed"):
                stage4.load_clean_raw()
        receipt = Path(config.ICA_REPORT_DIR) / self.manifest["model_id"] / "applied_components.json"
        record = stage4.read_json(receipt)
        record["output_sha256"] = "stale"
        stage3.write_json(receipt, record)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            stage4.load_clean_raw()

    def test_postica_settings_and_legacy_publication_rejected(self):
        with patch.object(config, "POSTICA_ABSOLUTE_AMPLITUDE_THRESHOLD_UV", 200.0):
            with self.assertRaisesRegex(ValueError, "Post-ICA amplitude settings"):
                stage4.load_clean_raw()
        receipt = Path(config.ICA_REPORT_DIR) / self.manifest["model_id"] / "applied_components.json"
        record = stage3.read_json(receipt)
        del record["amplitude_recheck"]
        stage3.write_json(receipt, record)
        with self.assertRaisesRegex(ValueError, "publication record"):
            stage4.load_clean_raw()

    def test_standalone_edge_rejects_without_mutating_input(self):
        raw = self.raw.copy()
        raw.annotations.append(13, .1, "edge_device_reset")
        before = raw.annotations.copy()
        epochs = stage4.make_epochs(raw)
        self.assertIn("bad_permanent_edge", epochs.drop_log[2])
        for key in ("onset", "duration", "description"):
            np.testing.assert_array_equal(getattr(before, key), getattr(raw.annotations, key))
        report = stage4.rejection_report(epochs, raw)
        self.assertEqual(report["bad_annotation_rejected_by_category"]["permanent"], 2)

    def test_invalid_reference_and_unknown_annotation_rejected(self):
        self.raw._data[0] += 1e-3
        self.raw.save(config.ICA_CLEAN_RAW_PATH, overwrite=True, fmt="double", verbose=False)
        with self.assertRaisesRegex(ValueError, "average reference"):
            stage4.load_clean_raw()
        self.raw._data[0] -= 1e-3
        self.raw.annotations.append(1, 0, "unknown_task")
        self.raw.save(config.ICA_CLEAN_RAW_PATH, overwrite=True, fmt="double", verbose=False)
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            stage4.load_clean_raw()

    def test_all_rejected_missing_class_and_duplicate_events(self):
        raw = self.raw.copy()
        raw.annotations.append(0, 64, "bad_amplitude")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(RuntimeError, "All stage 4"):
                stage4.make_epochs(raw)
        raw.set_annotations(mne.Annotations([5, 13, 12], [0, 0, 6],
                                           ["left", "right", "bad_amplitude"]))
        with self.assertRaisesRegex(RuntimeError, "class disappeared"):
            stage4.make_epochs(raw)
        raw.set_annotations(mne.Annotations([5, 5], [0, 0], ["left", "right"]))
        with self.assertRaises(RuntimeError):
            stage4.make_epochs(raw)

    def test_config_and_overwrite_protection(self):
        for key, value in (("STAGE4_EVENT_ID", {"left": 1, "right": 1}),
                           ("STAGE4_BASELINE", (-2.0, 0.0)),
                           ("STAGE4_OUTPUT_UNIT", "invalid-unit"),
                           ("STAGE4_FILTER_SKIP_ANNOTATIONS", ("edge",)),
                           ("STAGE4_INPUT_RAW_PATH", config.ICA_INPUT_RAW_PATH),
                           ("STAGE4_FULL_OUTPUT_PATH", config.STAGE4_WS_OUTPUT_PATH)):
            with self.subTest(key=key), patch.object(config, key, value):
                with self.assertRaises(ValueError):
                    stage4.validate_config()
        self.run_stage4()
        before = self.output_bytes()
        with patch.object(config, "STAGE4_OVERWRITE", False):
            with self.assertRaises(FileExistsError):
                stage4.main()
        self.assertEqual(before, self.output_bytes())

    def test_verification_failure_preserves_old_outputs(self):
        self.run_stage4()
        before = self.output_bytes()
        with patch.object(stage4, "verify_hdf5", side_effect=RuntimeError("readback failed")):
            with self.assertRaisesRegex(RuntimeError, "readback failed"):
                self.run_stage4()
        self.assertEqual(before, self.output_bytes())

    def test_each_replace_failure_rolls_back_all_outputs(self):
        self.run_stage4()
        before = self.output_bytes()
        replace = stage4.os.replace
        for target in stage4.output_paths():
            def fail_once(src, dst):
                if Path(dst) == target and Path(src).name != "previous.backup":
                    raise OSError("simulated publish failure")
                return replace(src, dst)
            with self.subTest(target=target), patch.object(stage4.os, "replace", side_effect=fail_once):
                with self.assertRaisesRegex(OSError, "simulated publish failure"):
                    self.run_stage4()
            self.assertEqual(before, self.output_bytes())

    def test_new_outputs_removed_on_failure(self):
        replace = stage4.os.replace
        def fail_report(src, dst):
            if Path(dst) == Path(config.STAGE4_REPORT_PATH):
                raise OSError("report publish failed")
            return replace(src, dst)
        with patch.object(stage4.os, "replace", side_effect=fail_report):
            with self.assertRaises(OSError):
                self.run_stage4()
        self.assertFalse(any(p.exists() for p in stage4.output_paths()))


if __name__ == "__main__":
    unittest.main(verbosity=2)