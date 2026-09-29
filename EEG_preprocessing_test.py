import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import mne
import numpy as np

import config
from EEG_preprocessing import (
    create_epochs_from_raw,
    detect_absolute_amplitude_intervals,
    detect_peak_change_intervals,
    preprocess_hdf5_file,
    concatenate_preprocessed_raws,
    save_concatenated_raw,
    validate_raw_list,
)


ACQUISITION_XML = """<?xml version="1.0" encoding="utf-8"?>
<AcquisitionTaskDescription>
  <ChannelProperties>
    <ChannelProperties><SampleRate>250</SampleRate><ChannelName>C3</ChannelName></ChannelProperties>
    <ChannelProperties><SampleRate>250</SampleRate><ChannelName>C4</ChannelName></ChannelProperties>
  </ChannelProperties>
</AcquisitionTaskDescription>
"""

EVENT_TYPES_XML = """<?xml version="1.0" encoding="utf-8"?>
<ArrayOfAsynchronSignalDescription>
  <AsynchronSignalDescription><ID>8</ID><Description>right</Description></AsynchronSignalDescription>
  <AsynchronSignalDescription><ID>9</ID><Description>left</Description></AsynchronSignalDescription>
</ArrayOfAsynchronSignalDescription>
"""


class EEGPreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "synthetic.hdf5"
        sfreq = 250.0
        times = np.arange(2500) / sfreq
        samples = np.column_stack(
            [20.0 * np.sin(2 * np.pi * 10 * times), 15.0 * np.sin(2 * np.pi * 12 * times)]
        ).astype(np.float32)
        samples[1498:1505, 0] = 700.0

        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(self.path, "w") as h5_file:
            raw_group = h5_file.create_group("RawData")
            raw_group.create_dataset("Samples", data=samples)
            raw_group.create_dataset(
                "AcquisitionTaskDescription", data=np.array([ACQUISITION_XML], dtype=string_dtype)
            )
            async_group = h5_file.create_group("AsynchronData")
            async_group.create_dataset("Time", data=np.array([[500], [1500]], dtype=np.int64))
            async_group.create_dataset("TypeID", data=np.array([[8], [9]], dtype=np.int32))
            async_group.create_dataset("Value", data=np.array([[1], [1]], dtype=np.int32))
            async_group.create_dataset(
                "AsynchronSignalTypes", data=np.array([EVENT_TYPES_XML], dtype=string_dtype)
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_annotations_filtering_and_epoch_rejection(self):
        raw = preprocess_hdf5_file(self.path)

        self.assertEqual(raw.ch_names, ["C3", "C4"])
        self.assertAlmostEqual(raw.info["sfreq"], 250.0)
        self.assertAlmostEqual(raw.info["highpass"], config.HIGH_PASS_FREQ)
        descriptions = list(raw.annotations.description)
        self.assertIn("right", descriptions)
        self.assertIn("left", descriptions)
        self.assertIn(config.PREICA_BAD_AMPLITUDE_LABEL, descriptions)
        self.assertTrue(detect_peak_change_intervals(raw))
        self.assertTrue(detect_absolute_amplitude_intervals(raw))
        self.assertTrue(
            all(
                description in {"right", "left", config.PREICA_BAD_AMPLITUDE_LABEL}
                for description in descriptions
            )
        )

        # 保留共同信号分量，证明没有执行平均参考（平均参考后逐时刻通道和接近 0）。
        channel_sum = raw.get_data(picks="eeg").sum(axis=0)
        self.assertGreater(float(np.max(np.abs(channel_sum))), 1e-8)

        epochs = create_epochs_from_raw(
            raw,
            tmin=-0.1,
            tmax=0.2,
            baseline=(None, 0.0),
        )
        self.assertEqual(len(epochs.drop_log), 2)
        self.assertEqual(len(epochs), 1)
        self.assertIn(config.PREICA_BAD_AMPLITUDE_LABEL, epochs.drop_log[1])


class RawConcatenationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output = Path(self.temp_dir.name) / "latest_raw.fif"
        self.raws = []
        for index in range(2):
            info = mne.create_info(["C3", "C4"], 250.0, "eeg")
            times = np.arange(1000) / 250.0
            data = np.tile(np.sin(2 * np.pi * 10 * times) * (index + 1) * 1e-5, (2, 1))
            raw = mne.io.RawArray(data, info, verbose=False)
            raw.filter(1.0, None, method="iir", verbose=False)
            raw.set_annotations(mne.Annotations([1.0, 2.0], [0, 0.1], ["left", "bad_amplitude"]))
            self.raws.append(raw)

    def test_concatenation_copies_inputs_and_normalizes_boundaries(self):
        original = [raw.get_data() for raw in self.raws]
        with patch("EEG_preprocessing.mne.concatenate_raws", wraps=mne.concatenate_raws) as concatenate:
            combined = concatenate_preprocessed_raws(self.raws, save_output=False)
        concatenate.assert_called_once()
        self.assertEqual(combined.n_times, 2000)
        self.assertEqual(list(combined.annotations.description).count("bad_boundary"), 1)
        self.assertNotIn("BAD boundary", combined.annotations.description)
        self.assertNotIn("EDGE boundary", combined.annotations.description)
        self.assertEqual(list(combined.annotations.description).count("left"), 2)
        self.assertEqual(list(combined.annotations.description).count("bad_amplitude"), 2)
        for raw, expected in zip(self.raws, original):
            np.testing.assert_array_equal(raw.get_data(), expected)
            self.assertEqual(len(raw.annotations), 2)
        # 拼接后已执行平均参考：逐采样点的 EEG 全局均值应为零。
        self.assertTrue(np.allclose(combined.get_data().mean(axis=0), 0.0, atol=1e-12))
        self.assertTrue(combined.info["custom_ref_applied"])
        self.assertFalse(combined.info["projs"])
        reference = __import__("json").loads(combined.info["description"])["eeg_reference"]
        self.assertIn("eeg_rank", reference)

    def test_single_input(self):
        combined = concatenate_preprocessed_raws(self.raws[:1], save_output=False)
        self.assertEqual(combined.n_times, 1000)
        self.assertNotIn("bad_boundary", combined.annotations.description)
        self.assertIsNot(combined, self.raws[0])

    def test_validation_rejects_mismatches(self):
        for change in ("channels", "sfreq", "highpass", "bads", "reference", "nan", "time"):
            with self.subTest(change=change):
                raws = [raw.copy() for raw in self.raws]
                if change == "channels":
                    raws[1].reorder_channels(["C4", "C3"])
                elif change == "sfreq":
                    raws[1].resample(200, verbose=False)
                elif change == "highpass":
                    raws[1].filter(2, None, method="iir", verbose=False)
                elif change == "bads":
                    raws[1].info["bads"] = ["C3"]
                elif change == "reference":
                    raws[1].set_eeg_reference("average", verbose=False)
                elif change == "nan":
                    raws[1].apply_function(lambda values: values * np.nan, verbose=False)
                elif change == "time":
                    raws[1].crop(tmin=0.1)
                with self.assertRaises(ValueError):
                    concatenate_preprocessed_raws(raws, save_output=False)
        with self.assertRaises(ValueError):
            validate_raw_list([])
        with self.assertRaises(TypeError):
            validate_raw_list([object()])
        with self.assertRaises(ValueError):
            validate_raw_list([self.raws[0], self.raws[0]])

    def test_reloaded_output_is_latest_successful_run(self):
        first = concatenate_preprocessed_raws(self.raws, output_path=self.output)
        self.assertTrue(self.output.exists())
        second = concatenate_preprocessed_raws(self.raws[:1], output_path=self.output)
        with mne.io.read_raw_fif(self.output, preload=True, verbose=False) as loaded:
            self.assertEqual(loaded.n_times, second.n_times)
            self.assertNotEqual(loaded.n_times, first.n_times)
            np.testing.assert_allclose(loaded.get_data(), second.get_data())
            self.assertEqual(list(loaded.annotations.description), list(second.annotations.description))
        self.assertEqual([path.name for path in self.output.parent.iterdir()], [self.output.name])

    def test_save_failures_keep_previous_result(self):
        combined = concatenate_preprocessed_raws(self.raws, output_path=self.output)
        previous = self.output.read_bytes()
        for target in ("mne.io.BaseRaw.save", "EEG_preprocessing.mne.io.read_raw_fif", "EEG_preprocessing.os.replace"):
            with self.subTest(target=target), patch(target, side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    save_concatenated_raw(combined, self.output)
            self.assertEqual(self.output.read_bytes(), previous)
            self.assertEqual([path.name for path in self.output.parent.iterdir()], [self.output.name])
        with patch.object(config, "CONCATENATED_RAW_OVERWRITE", False):
            with self.assertRaises(FileExistsError):
                save_concatenated_raw(combined, self.output)

    def test_split_output_is_not_published(self):
        combined = concatenate_preprocessed_raws(self.raws, output_path=self.output)
        previous = self.output.read_bytes()

        def fake_split_save(path, **kwargs):
            Path(path).write_bytes(b"part one")
            Path(path).with_name("stage2_raw-1.fif").write_bytes(b"part two")

        with patch.object(combined, "save", side_effect=fake_split_save):
            with self.assertRaisesRegex(RuntimeError, "split FIF"):
                save_concatenated_raw(combined, self.output)
        self.assertEqual(self.output.read_bytes(), previous)

    def test_boundary_rejects_crossing_epoch(self):
        combined = concatenate_preprocessed_raws(self.raws, save_output=False)
        # 边界由 MNE 产生，测试直接使用其位置，不手算累计时间。
        boundary = next(a["onset"] for a in combined.annotations if a["description"] == "bad_boundary")
        combined.annotations.append(boundary - 0.04, 0, "crossing")
        events, event_id = mne.events_from_annotations(combined, event_id={"crossing": 10}, verbose=False)
        with self.assertWarns(RuntimeWarning):
            epochs = mne.Epochs(combined, events, event_id, tmin=-0.1, tmax=0.2,
                                baseline=None, preload=True, reject_by_annotation=True, verbose=False)
        self.assertEqual(len(epochs), 0)
        self.assertIn("bad_boundary", epochs.drop_log[0])


if __name__ == "__main__":
    unittest.main()