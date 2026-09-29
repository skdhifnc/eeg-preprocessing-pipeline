"""阶段 3 合成数据测试；不触碰正式模型，不进行真实数据人工伪迹判定。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

import config
import ICA_processing as stage3


class ICAProcessingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        settings = dict(
            ICA_INPUT_RAW_PATH=self.root / "input_raw.fif",
            ICA_MODEL_PATH=self.root / "model-ica.fif",
            ICA_MANIFEST_PATH=self.root / "manifest.json",
            ICA_CLEAN_RAW_PATH=self.root / "clean_raw.fif",
            # 四通道即时平均参考后最多只有三维独立子空间。
            ICA_REPORT_DIR=self.root / "review", ICA_N_COMPONENTS=3,
            ICA_MAX_ITER=1000, ICA_BOUNDARY_PADDING_SECONDS=0.1,
            ICA_EXCLUDE_COMPONENTS=(), ICA_SOURCE_WINDOW_SECONDS=2.0,
            ICA_PSD_FMAX=40.0, ICA_PREVIEW_CHANNELS=("Fp1", "Fp2", "C3", "C4"),
            POSTICA_AMPLITUDE_PEAK_UV=100.0, POSTICA_AMPLITUDE_PEAK_MIN_DURATION=0.004,
            POSTICA_ABSOLUTE_AMPLITUDE_THRESHOLD_UV=300.0,
            POSTICA_ABSOLUTE_AMPLITUDE_MIN_DURATION=0.004, POSTICA_AMPLITUDE_MERGE_GAP_SECONDS=0.0,
        )
        for name, value in settings.items():
            context = patch.object(config, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.addCleanup(plt.close, "all")
        rng = np.random.default_rng(20)
        source = rng.laplace(size=(4, 4000))
        mixing = rng.normal(size=(4, 4))
        data = mixing @ source * 1e-5
        info = mne.create_info(["Fp1", "Fp2", "C3", "C4"], 100.0, "eeg")
        self.raw = mne.io.RawArray(data, info, verbose=False)
        self.raw.set_montage(config.ICA_MONTAGE, on_missing="raise")
        self.raw.filter(1, None, method="iir", verbose=False)
        self.raw.set_annotations(mne.Annotations([2, 5, 10, 25], [0, 0.5, 0, 0],
                                 ["left", "bad_amplitude", "bad_boundary", "right"]))
        self.raw.set_eeg_reference("average", projection=False, verbose=False)
        self.raw.info["description"] = __import__("json").dumps({
            "stage": 2,
            "eeg_reference": {
                "schema_version": 1,
                "reference": "average",
                "projection": False,
                "channel_names": self.raw.ch_names,
                "n_times": int(self.raw.n_times),
                "eeg_rank": 3,
            },
        })
        self.raw.save(config.ICA_INPUT_RAW_PATH, fmt="double", verbose=False)

    def fit(self):
        return stage3.fit_stage()

    def test_mapping_and_no_reference(self):
        info = mne.create_info(["CZ"] + [str(i) for i in range(2, 33)], 250, "eeg")
        raw = mne.io.RawArray(np.zeros((32, 100)), info, verbose=False)
        prepared, mapping = stage3.prepare_channels(raw)
        self.assertEqual(mapping["CZ"], "Fp1")
        self.assertEqual(prepared.ch_names, list(stage3.DEVICE_CHANNELS))
        self.assertFalse(prepared.info["custom_ref_applied"])
        self.assertEqual(raw.ch_names[0], "CZ")
        again, _ = stage3.prepare_channels(prepared)
        self.assertEqual(again.ch_names, prepared.ch_names)
        raw.rename_channels({"2": "unknown"})
        with self.assertRaises(ValueError):
            stage3.prepare_channels(raw)

    def test_fit_omits_amplitude_and_padded_boundary_and_reuses(self):
        before = stage3.file_hash(config.ICA_INPUT_RAW_PATH)
        ica, manifest = self.fit()
        raw, _, _ = stage3.load_matching()
        fit_raw = stage3.make_fit_raw(raw)
        expected = fit_raw.get_data(reject_by_annotation="omit").shape[1]
        self.assertEqual(ica.n_samples_, expected)
        self.assertLess(expected, raw.n_times - 50)  # 幅值段 + 接缝窗口
        self.assertEqual(raw.annotations.duration[2], 0)
        self.assertGreater(fit_raw.annotations.duration[2], 0)
        self.assertEqual(stage3.file_hash(config.ICA_INPUT_RAW_PATH), before)
        _, reused = self.fit()
        self.assertEqual(reused["model_id"], manifest["model_id"])

    def test_inspect_preview_apply_and_failure_preserves_output(self):
        self.fit()
        stage3.inspect_stage([0], show=False)
        self.assertFalse(Path(config.ICA_CLEAN_RAW_PATH).exists())
        raw, ica, manifest = stage3.load_matching()
        model_hash = stage3.file_hash(config.ICA_MODEL_PATH)
        with patch.object(config, "ICA_EXCLUDE_COMPONENTS", (0,)):
            clean = stage3.preview_stage(show=False)
            self.assertFalse(Path(config.ICA_CLEAN_RAW_PATH).exists())
            self.assertFalse(np.allclose(clean.get_data(), raw.get_data(), atol=1e-12))
            self.assertEqual(stage3.file_hash(config.ICA_MODEL_PATH), model_hash)
            result = stage3.apply_stage(confirm=True)
            with mne.io.read_raw_fif(config.ICA_CLEAN_RAW_PATH, preload=True, verbose=False) as loaded:
                np.testing.assert_allclose(loaded.get_data(), result.get_data(), atol=1e-14)
                self.assertEqual(list(loaded.annotations.description), list(clean.annotations.description))
                self.assertEqual(loaded.n_times, raw.n_times)
            receipt = stage3.read_json(stage3.report_dir(manifest) / "applied_components.json")
            self.assertEqual(receipt["amplitude_recheck"]["pre_ica"]["count"], 1)
            with patch.object(config, "POSTICA_ABSOLUTE_AMPLITUDE_THRESHOLD_UV", 200.0):
                with self.assertRaisesRegex(RuntimeError, "preview again"):
                    stage3.apply_stage(confirm=True)
            previous = Path(config.ICA_CLEAN_RAW_PATH).read_bytes()
            with patch("ICA_processing.os.replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    stage3.apply_stage(confirm=True)
            self.assertEqual(Path(config.ICA_CLEAN_RAW_PATH).read_bytes(), previous)
        with patch.object(config, "ICA_RANDOM_STATE", 7):
            with self.assertRaisesRegex((RuntimeError, ValueError), "stale|settings|average reference"):
                stage3.load_matching()
        folder = stage3.report_dir(manifest)
        self.assertTrue((folder / "ICA000_properties.png").is_file())
        self.assertTrue((folder / "before_after.png").is_file())
    def test_stale_input_settings_model_rejected(self):
        self.fit()
        with patch.object(config, "ICA_RANDOM_STATE", 7):
            with self.assertRaisesRegex((RuntimeError, ValueError), "stale|settings|average reference"):
                stage3.load_matching()
        model_path = Path(config.ICA_MODEL_PATH)
        original = model_path.read_bytes()
        model_path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            stage3.load_matching()
        model_path.write_bytes(original)
        self.raw.info["description"] = "new stage 2 run"
        self.raw.save(config.ICA_INPUT_RAW_PATH, fmt="double", overwrite=True, verbose=False)
        with self.assertRaisesRegex((RuntimeError, ValueError), "stale|average reference"):
            stage3.load_matching()

    def test_invalid_selection_and_confirmation(self):
        for values in ([True], [1.5], [-1], [4], [0, 0], [0, 1, 2, 3]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                stage3.validate_exclusions(values, 4)
        with self.assertRaises(ValueError):
            stage3.apply_stage(confirm=False)
        self.fit()
        with self.assertRaisesRegex(RuntimeError, "Run preview"):
            stage3.apply_stage(confirm=True)

    def test_reconstruction_keeps_original_and_annotations(self):
        self.fit()
        raw, ica, _ = stage3.load_matching()
        before = raw.get_data()
        with self.assertWarns(RuntimeWarning):
            unchanged = stage3.reconstruct(raw, ica, [])
        np.testing.assert_allclose(unchanged.get_data(), before, atol=1e-12)
        stage3.reconstruct(raw, ica, [0])
        np.testing.assert_array_equal(raw.get_data(), before)
        self.assertEqual(ica.exclude, [])


    def test_button_review_does_not_request_terminal_input(self):
        self.fit()
        with patch.object(stage3, "overview"), \
                patch.object(mne.preprocessing.ICA, "plot_sources"), \
                patch("builtins.input", side_effect=AssertionError("Unexpected terminal input")):
            self.assertEqual(stage3.inspect_stage(show=True, prompt=False), [])


class AnnotationRoundTripTest(unittest.TestCase):
    """FIF 可重排同 onset 注释，但不能改变完整注释记录。"""

    def test_same_onset_reordering_and_fif_quantization_are_accepted(self):
        original = mne.Annotations(
            [10, 10, 20], [0, .22, .38], ["left", "bad_amplitude", "bad_clipping"]
        )
        restored = mne.Annotations(
            [10.000001, 20.000002, 10.000001],
            [.220001, .379997, 0],
            ["bad_amplitude", "bad_clipping", "left"],
        )
        stage3.validate_annotation_roundtrip(original, restored, 250.0)

    def test_real_annotation_changes_are_rejected(self):
        original = mne.Annotations([10, 10], [0, .22], ["left", "bad_amplitude"])
        cases = [
            mne.Annotations([10, 10], [0, .3], ["left", "bad_amplitude"]),
            mne.Annotations([10, 10], [0, .22], ["right", "bad_amplitude"]),
            mne.Annotations([10], [0], ["left"]),
        ]
        for changed in cases:
            with self.subTest(changed=list(changed.description)), self.assertRaises(RuntimeError):
                stage3.validate_annotation_roundtrip(original, changed, 250.0)

    def test_channel_scopes_are_part_of_annotation_identity(self):
        original = mne.Annotations([1], [.1], ["bad_amplitude"], ch_names=[["C3"]])
        changed = mne.Annotations([1], [.1], ["bad_amplitude"], ch_names=[["C4"]])
        with self.assertRaisesRegex(RuntimeError, "names/channel scopes"):
            stage3.validate_annotation_roundtrip(original, changed, 250.0)


class AmplitudeRecheckTest(unittest.TestCase):
    """确定性的 clean 信号复检：不把合成 ICA 成分选择当成真实数据审核。"""

    def setUp(self):
        context = patch.multiple(
            config, POSTICA_AMPLITUDE_PEAK_UV=100.0,
            POSTICA_AMPLITUDE_PEAK_MIN_DURATION=0.004,
            POSTICA_ABSOLUTE_AMPLITUDE_THRESHOLD_UV=300.0,
            POSTICA_ABSOLUTE_AMPLITUDE_MIN_DURATION=0.004,
            POSTICA_AMPLITUDE_MERGE_GAP_SECONDS=0.0,
        )
        context.start()
        self.addCleanup(context.stop)
        self.raw = mne.io.RawArray(np.zeros((2, 2500)), mne.create_info(["C3", "C4"], 250, "eeg"),
                                   verbose=False)
        self.raw.set_annotations(mne.Annotations(
            [1, 1, 3, 3, 5, 5, 7, 7], [0, .1, 0, .1, 0, .1, 0, .1],
            ["left", "bad_preica_amplitude", "right", "bad_amplitude",
             "left", "bad_clipping", "right", "bad_amplitude_hardware"]))

    def test_recovered_epochs_return_and_persistent_artifacts_reject(self):
        self.raw._data[:, 750:775] = np.array([[500e-6], [-500e-6]])
        before = self.raw.get_data().copy()
        audit = stage3.recheck_amplitude(self.raw)
        np.testing.assert_array_equal(before, self.raw.get_data())
        self.assertNotIn("bad_preica_amplitude", self.raw.annotations.description)
        self.assertEqual(audit["pre_ica"]["count"], 2)
        self.assertAlmostEqual(audit["pre_ica"]["duration_seconds"], .2)
        self.assertEqual(audit["post_ica"]["count"], 1)
        self.assertEqual(audit["permanent"]["count"], 2)
        events, ids = mne.events_from_annotations(self.raw, event_id={"left": 1, "right": 2}, verbose=False)
        epochs = mne.Epochs(self.raw, events, ids, tmin=0, tmax=.2, baseline=None,
                            preload=True, reject_by_annotation=True, verbose=False)
        self.assertEqual(epochs.selection.tolist(), [0])
        self.assertIn("bad_amplitude", epochs.drop_log[1])
        self.assertIn("bad_clipping", epochs.drop_log[2])
        self.assertIn("bad_amplitude_hardware", epochs.drop_log[3])

    def test_all_permanent_annotations_preserved_exactly(self):
        names = ["bad_boundary", "bad_acq_skip", "edge", "bad_dropout", "bad_flat", "bad_clipping"]
        self.raw.set_annotations(mne.Annotations(np.arange(1, 7), [0, .1, 0, .2, .3, .4], names))
        before = self.raw.annotations.copy()
        audit = stage3.recheck_amplitude(self.raw)
        for key in ("onset", "duration", "description"):
            np.testing.assert_array_equal(getattr(before, key), getattr(self.raw.annotations, key))
        self.assertEqual(audit["post_ica"]["count"], 0)
        self.assertEqual(audit["permanent"]["count"], 6)

    def test_postica_threshold_duration_and_merge_are_independent(self):
        self.raw._data[:, 250:260] = 200e-6
        self.raw._data[:, 262:272] = 200e-6
        with patch.multiple(config, POSTICA_AMPLITUDE_PEAK_UV=1000.0,
                            POSTICA_ABSOLUTE_AMPLITUDE_THRESHOLD_UV=150.0,
                            POSTICA_ABSOLUTE_AMPLITUDE_MIN_DURATION=.02,
                            POSTICA_AMPLITUDE_MERGE_GAP_SECONDS=.01,
                            ABSOLUTE_AMPLITUDE_THRESHOLD_UV=10000.0):
            audit = stage3.recheck_amplitude(self.raw)
            self.assertEqual(audit["post_ica"]["count"], 1)
            self.assertAlmostEqual(audit["post_ica"]["duration_seconds"], .088)
            with patch.object(config, "POSTICA_ABSOLUTE_AMPLITUDE_MIN_DURATION", .05):
                self.assertEqual(stage3.recheck_amplitude(self.raw)["post_ica"]["count"], 0)

    def test_invalid_settings_or_detection_failure_keep_annotations(self):
        before = self.raw.annotations.copy()
        for value in (0, -1, float("nan"), True):
            with self.subTest(value=value), patch.object(config, "POSTICA_AMPLITUDE_PEAK_UV", value):
                with self.assertRaises(ValueError):
                    stage3.recheck_amplitude(self.raw)
        with patch.object(stage3, "annotate_bad_amplitude", side_effect=RuntimeError("detection failed")):
            with self.assertRaises(RuntimeError):
                stage3.recheck_amplitude(self.raw)
        for key in ("onset", "duration", "description"):
            np.testing.assert_array_equal(getattr(before, key), getattr(self.raw.annotations, key))


class ICARunInterfaceTest(unittest.TestCase):
    """入口调度测试不读取真实 EEG，也不发布正式文件。"""

    def setUp(self):
        context = patch.multiple(
            config, ICA_RUN_MODE="review", ICA_FORCE_REFIT=False,
            ICA_SHOW_PLOTS=True, ICA_REVIEW_COMPONENTS=None, ICA_APPLY_CONFIRM=False,
        )
        context.start()
        self.addCleanup(context.stop)
        backend = patch.object(stage3.plt, "switch_backend")
        self.backend = backend.start()
        self.addCleanup(backend.stop)

    def test_config_dispatches_all_modes(self):
        for mode, target, positional, keywords in (
            ("fit", "fit_stage", (False,), {}),
            ("review", "inspect_stage", (None, True), {"prompt": False}),
            ("inspect", "inspect_stage", (None, True), {"prompt": False}),
            ("preview", "preview_stage", (True,), {}),
            ("apply", "apply_stage", (False,), {}),
        ):
            with self.subTest(mode=mode), patch.object(config, "ICA_RUN_MODE", mode), \
                    patch.object(stage3, target) as handler:
                self.assertIs(stage3.main([]), handler.return_value)
                handler.assert_called_once_with(*positional, **keywords)

    def test_config_options_and_sys_argv(self):
        with patch.multiple(config, ICA_REVIEW_COMPONENTS=(0, 2), ICA_SHOW_PLOTS=False), \
                patch.object(stage3.sys, "argv", [str(Path(stage3.__file__))]), \
                patch.object(stage3, "inspect_stage") as handler:
            stage3.main()
            handler.assert_called_once_with((0, 2), False, prompt=False)
            self.backend.assert_called_once_with("Agg")
        with patch.multiple(config, ICA_RUN_MODE="fit", ICA_FORCE_REFIT=True), \
                patch.object(stage3, "fit_stage") as handler:
            stage3.main([])
            handler.assert_called_once_with(True)
        with patch.multiple(config, ICA_RUN_MODE="apply", ICA_APPLY_CONFIRM=True), \
                patch.object(stage3, "apply_stage") as handler:
            stage3.main([])
            handler.assert_called_once_with(True)

    def test_config_publish_requires_explicit_confirmation(self):
        with patch.object(config, "ICA_RUN_MODE", "apply"), \
                patch.object(stage3, "load_matching") as load:
            with self.assertRaisesRegex(ValueError, "ICA_APPLY_CONFIRM"):
                stage3.main([])
            load.assert_not_called()

    def test_invalid_config_rejected_before_dispatch(self):
        for name, value in (("ICA_RUN_MODE", "typo"), ("ICA_RUN_MODE", []),
                            ("ICA_FORCE_REFIT", "False"), ("ICA_SHOW_PLOTS", 1),
                            ("ICA_APPLY_CONFIRM", "True")):
            with self.subTest(name=name, value=value), patch.object(config, name, value):
                with self.assertRaisesRegex(ValueError, name):
                    stage3.main([])

    def test_cli_overrides_run_config(self):
        with patch.multiple(config, ICA_RUN_MODE="invalid", ICA_APPLY_CONFIRM=True), \
                patch.object(stage3, "apply_stage") as handler:
            stage3.main(["apply"])
            handler.assert_called_once_with(False)
        with patch.object(stage3, "inspect_stage") as handler:
            stage3.main(["review", "--components", "0", "2", "--no-show"])
            handler.assert_called_once_with([0, 2], False, prompt=True)
        with patch.object(stage3, "fit_stage") as handler:
            stage3.main(["fit", "--force"])
            handler.assert_called_once_with(True)
        with patch.object(stage3, "apply_stage") as handler:
            stage3.main(["apply", "--confirm"])
            handler.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()