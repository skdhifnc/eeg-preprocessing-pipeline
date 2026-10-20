"""复用现有阶段 3 发布夹具，验证阶段4独立 FBCSP 导出及回滚。"""
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import config
import EEG_processing_test as fixtures
import EEG_processing as stage4
import fbcsp
import fbcsp_main
import fbcsp_stage4 as fb4


class FBCSPStage4Test(unittest.TestCase):
    def setUp(self):
        fixtures.EpochProcessingTest.setUp(self)
        settings = dict(FBCSP_FULL_OUTPUT_PATH=self.root / "fb_full.hdf5",
                        FBCSP_WS_OUTPUT_PATH=self.root / "fb_ws.hdf5",
                        FBCSP_BASELINE_OUTPUT_PATH=self.root / "fb_base.hdf5",
                        FBCSP_REPORT_PATH=self.root / "fb_report.json", FBCSP_OVERWRITE=True)
        for key, value in settings.items():
            context = patch.object(config, key, value)
            context.start()
            self.addCleanup(context.stop)

    def run_export(self):
        with redirect_stdout(StringIO()):
            return fb4.main()

    def test_real_export_alignment_and_original_files_untouched(self):
        before = fbcsp.file_sha256(config.ICA_CLEAN_RAW_PATH)
        config.STAGE4_WS_OUTPUT_PATH.write_bytes(b"historical CSP input")
        full, cropped, report = self.run_export()
        self.assertEqual(full[0].shape, (5, 5, 32, 1251))
        self.assertEqual(cropped[0].shape, (5, 5, 32, 1000))
        np.testing.assert_array_equal(cropped[0], full[0][:, :, :, 250:1250])
        np.testing.assert_array_equal(cropped[1], [2, 1, 1, 2, 1])
        fbcsp_main.load_publication(config.FBCSP_REPORT_PATH)
        self.assertEqual(config.STAGE4_WS_OUTPUT_PATH.read_bytes(), b"historical CSP input")
        self.assertEqual(fbcsp.file_sha256(config.ICA_CLEAN_RAW_PATH), before)
        self.assertEqual(report["rejection"]["accepted_epochs"], 5)
        with h5py.File(config.FBCSP_BASELINE_OUTPUT_PATH, "r") as base:
            clean, _ = stage4.load_clean_raw()
            expected = stage4.arrays(stage4.make_epochs(stage4.filter_raw(clean)).copy().crop(0, 3.996))
            np.testing.assert_array_equal(base["clean_eeg"][:], expected[0])
            np.testing.assert_array_equal(base["labels"][:], cropped[1])
            np.testing.assert_array_equal(base["event_samples"][:], cropped[4])

    def test_publish_rollback_on_replace_failure(self):
        self.run_export()
        _, paths = fb4.validate_fbcsp_config()
        previous = {p: p.read_bytes() for p in paths}
        replace = fb4.os.replace
        count = 0

        def fail_once(src, dst):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("synthetic publication failure")
            return replace(src, dst)

        with patch.object(fb4.os, "replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "synthetic"):
                self.run_export()
        for p, contents in previous.items():
            self.assertEqual(p.read_bytes(), contents)

    def test_invalid_bands_overwrite_and_source_changes(self):
        with patch.object(config, "FBCSP_BANDS", ((8, 200),)):
            with self.assertRaises(ValueError):
                fb4.validate_fbcsp_config(250)
        self.run_export()
        with patch.object(config, "FBCSP_OVERWRITE", False):
            with self.assertRaises(FileExistsError):
                self.run_export()
        original = stage4.verify_source
        count = 0

        def changed(source):
            nonlocal count
            count += 1
            if count > 1:
                raise RuntimeError("source changed")
            return original(source)

        _, paths = fb4.validate_fbcsp_config()
        before = {p: p.read_bytes() for p in paths}
        with patch.object(stage4, "verify_source", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                self.run_export()
        for p in paths:
            self.assertEqual(before[p], p.read_bytes())

    def test_band_response_and_boundary_isolation(self):
        import mne
        raw = self.raw.copy()
        t = raw.times
        signal = sum(np.sin(2 * np.pi * frequency * t) for frequency in (3, 10, 25)) * 1e-5
        raw._data[:] = 0
        raw._data[0] = signal
        raw._data[1] = -signal
        raw.set_annotations(mne.Annotations([], [], []))
        before = raw.get_data().copy()
        filtered = fb4.filter_band(raw, 8., 12.)
        np.testing.assert_array_equal(raw.get_data(), before)
        data, time = filtered.get_data()[0, 2500:12500], t[2500:12500]
        amplitude = [abs(np.dot(data, np.exp(-2j * np.pi * f * time))) * 2 / len(time)
                     for f in (3, 10, 25)]
        self.assertGreater(amplitude[1], 9e-6)
        self.assertLess(amplitude[0], 1e-7)
        self.assertLess(amplitude[2], 1e-7)
        raw._data[:] = 0
        raw._data[0, 8000:] = signal[8000:]
        raw._data[1, 8000:] = -signal[8000:]
        raw.set_annotations(mne.Annotations([32.], [0.], [config.BAD_BOUNDARY_LABEL]))
        isolated = fb4.filter_band(raw, 8., 12.)
        np.testing.assert_array_equal(isolated.get_data()[:, :8000], 0.)


if __name__ == "__main__":
    unittest.main()