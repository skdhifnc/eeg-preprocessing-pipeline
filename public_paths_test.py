"""Check public configuration and auxiliary inputs using synthetic data only."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import tempfile
import types
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parent


class PublicPathsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.config = types.ModuleType("config")
        self.config.__dict__.update(runpy.run_path(str(ROOT / "config.example.py")))
        self.raw_path = self.folder / "synthetic.hdf5"
        self.config.ACQUISITION_VIEW_INPUT_PATH = self.raw_path
        self.config.HDF5_VIEW_INPUT_PATH = self.raw_path
        self.config.MAT_DATASET_DIR = self.folder
        self.config.MAT_INPUT_PATH = self.folder / "dataset1_processeddata_Sub-001_sess-01_task-MI_eeg.mat"
        self.config.MAT_OUTPUT_PATH = self.folder / "clean_dataset1_processeddata_Sub-001_sess-01_task-MI_eeg.hdf5"

    def run_script(self, name, arguments=()):
        previous = Path.cwd()
        try:
            os.chdir(self.folder)
            with patch.dict("sys.modules", {"config": self.config}), \
                    patch("sys.argv", [name, *arguments]), \
                    contextlib.redirect_stdout(io.StringIO()):
                return runpy.run_path(str(ROOT / name), run_name="__main__")
        finally:
            os.chdir(previous)

    def test_template_is_complete_and_safe(self):
        self.assertEqual(self.config.DATASET_DIR, ROOT / "data")
        self.assertEqual(self.config.get_input_indices(), tuple(range(1, 11)))
        self.assertEqual(self.config.ICA_EXCLUDE_COMPONENTS, ())
        self.assertFalse(self.config.ICA_APPLY_CONFIRM)
        self.assertEqual(self.config.ICA_RUN_MODE, "fit")
        self.assertEqual(self.config.MAT_BATCH_START_SUB, 1)
        self.assertEqual(self.config.MAT_BATCH_END_SUB, 1)
        self.assertEqual(self.config.MAT_BATCH_N_SESSIONS, 1)

    def test_viewers_use_local_config_and_export_xml(self):
        with h5py.File(self.raw_path, "w") as stream:
            stream.create_dataset("RawData/Samples", data=np.zeros((10, 2)))
            stream.create_dataset("RawData/SessionDescription", data=[b"synthetic"])
            stream.create_dataset("RawData/AcquisitionTaskDescription", data=[b"<Acquisition />"])
            stream.create_dataset("AsynchronData/Time", data=[1])
            stream.create_dataset("AsynchronData/TypeID", data=[8])
            stream.create_dataset("AsynchronData/AsynchronSignalTypes", data=[b"<Events />"])
        for name in ("Core_EEG_data_acquisition.py", "test.py",
                     "Viewing_HDF5_Structure.py", "XML_Metadata_Viewer.py"):
            with self.subTest(script=name):
                self.run_script(name)
        self.assertEqual((self.folder / "acquisition_desc.xml").read_text(encoding="utf-8"), "<Acquisition />")
        self.assertEqual((self.folder / "asynchron_signal_types.xml").read_text(encoding="utf-8"), "<Events />")

    def make_mat(self):
        sio.savemat(self.config.MAT_INPUT_PATH, {
            "data": np.zeros((58, 1000, 200), dtype=np.float32),
            "labels": np.tile(np.array([[1, 2]], dtype=np.uint8), (1, 100)),
        }, do_compression=True)

    def verify_output(self):
        with h5py.File(self.config.MAT_OUTPUT_PATH, "r") as stream:
            self.assertEqual(stream["clean_eeg"].shape, (200, 58, 1000))
            self.assertEqual(stream["labels"].shape, (200,))
            self.assertEqual(stream.attrs["source_mat_path"], str(self.config.MAT_INPUT_PATH))

    def test_single_conversion_preserves_configured_paths(self):
        self.make_mat()
        self.run_script("convert_mat_to_clean_hdf5.py", ["--dry-run"])
        self.assertFalse(self.config.MAT_OUTPUT_PATH.exists())
        self.run_script("convert_mat_to_clean_hdf5.py")
        self.verify_output()

    def test_batch_conversion_preserves_configured_paths(self):
        self.make_mat()
        self.run_script("convert_mat_to_clean_hdf5_test.py", ["--dry-run"])
        self.assertFalse(self.config.MAT_OUTPUT_PATH.exists())
        self.run_script("convert_mat_to_clean_hdf5_test.py")
        self.verify_output()

    def test_invalid_batch_range_is_rejected(self):
        self.config.MAT_BATCH_END_SUB = 0
        with self.assertRaisesRegex(ValueError, "Invalid MAT batch"):
            self.run_script("convert_mat_to_clean_hdf5_test.py", ["--dry-run"])


if __name__ == "__main__":
    unittest.main()