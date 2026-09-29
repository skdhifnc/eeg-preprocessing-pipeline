"""只用临时合成数据测试置换，不启动正式 EEG 置换、不修改既有实验结果。"""
from contextlib import redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

import h5py
import mne
import numpy as np
from scipy.io import loadmat, savemat

import main as classification
import permutation_test as permutation
import ws
from ws_test import NestedCVTest as _Fixture


class PermutationTest(unittest.TestCase):
    write_data = _Fixture.write_data
    _fixture_setup = _Fixture.setUp

    def setUp(self):
        self._fixture_setup()
        self.mat_path = self.root / "observed.mat"
        self.txt_path = self.root / "observed.txt"
        self.grid["svm__C"] = [0.1, 1.0]
        with mne.utils.use_log_level("ERROR"), redirect_stdout(StringIO()):
            classification.run_classification(clean_data_path=self.path,
                                              result_mat_path=self.mat_path,
                                              result_txt_path=self.txt_path, param_grid=self.grid)

    def run_permutations(self, **kwargs):
        kwargs.setdefault("output_dir", self.root / "permutations")
        kwargs.setdefault("n_permutations", 2)
        with mne.utils.use_log_level("ERROR"), redirect_stdout(StringIO()):
            return permutation.run_permutation_test(observed_mat_path=self.mat_path, **kwargs)

    def test_labels_only_full_nested_refits_and_output_round_trip(self):
        protected = {path: path.read_bytes() for path in (self.path, self.mat_path, self.txt_path)}
        nested = ws.nested_cv
        captured, fits = [], []
        original_fit = ws.CSP.fit

        def tracked_nested(X, y, **kwargs):
            np.testing.assert_array_equal(X, self.data)
            self.assertFalse(X.flags.writeable)
            np.testing.assert_array_equal(np.sort(y), np.sort(self.labels))
            self.assertEqual(kwargs["param_grid"], self.grid)
            self.assertEqual((kwargs["k_fold"], kwargs["inner_k_fold"]), (10, 5))
            captured.append(y.copy())
            return nested(X, y, **kwargs)

        def tracked_fit(estimator, X, y):
            fits.append(estimator)
            return original_fit(estimator, X, y)

        with patch.object(ws, "nested_cv", side_effect=tracked_nested), \
                patch.object(ws.CSP, "fit", tracked_fit):
            result = self.run_permutations()
        # 同一内层折两种 C 共用 CSP；每次 10*(5+1)=60，置换间重新拟合。
        self.assertEqual(len(fits), 120)
        self.assertEqual(len({id(estimator) for estimator in fits}), 120)
        self.assertEqual(len(captured), 2)
        rng = np.random.default_rng(42)
        for i, item in enumerate(result["permutations"]):
            indices = rng.permutation(len(self.labels))
            self.assertEqual(item["permutation_indices"], indices.tolist())
            np.testing.assert_array_equal(captured[i], self.labels[indices])
            self.assertEqual(item["permuted_labels"], self.labels[indices].tolist())
            self.assertEqual(len(item["cv_details"]["folds"]), 10)
        self.assertFalse(np.array_equal(captured[0], self.labels))
        self.assertEqual(result["status"], "completed")
        p_value, count = permutation.empirical_p_value(
            result["observed_score"], [item["accuracy"] for item in result["permutations"]])
        self.assertEqual((result["p_value"], result["n_extreme"]), (p_value, count))
        directory = self.root / "permutations"
        self.assertEqual(json.loads((directory / "permutation_results.json").read_text(encoding="utf-8")), result)
        stored = loadmat(directory / "permutation_summary.mat", simplify_cells=True)
        self.assertEqual(stored["p_value"], p_value)
        self.assertEqual(stored["permutation_fold_scores"].shape, (2, 10))
        self.assertIn("Empirical P value:", (directory / "permutation_summary.txt").read_text())
        for path, before in protected.items():
            self.assertEqual(path.read_bytes(), before)

    def test_seed_reproducible(self):
        first = self.run_permutations()
        second = self.run_permutations(output_dir=self.root / "repeat")
        self.assertEqual(first["permutations"], second["permutations"])
        self.assertEqual(first["p_value"], second["p_value"])

    def test_resume_matches_uninterrupted_and_rejects_corruption(self):
        # 旧reference实验中断后，用新grouped引擎继续，分数/最优参数也必须一致。
        full = self.run_permutations(output_dir=self.root / "full", search_engine="reference")
        directory = self.root / "interrupted"
        directory.mkdir()
        checkpoint = directory / "permutation_results.json"
        partial = json.loads(json.dumps(full))
        partial.update(status="failed", n_completed=1, p_value=None, n_extreme=None)
        partial["permutations"] = partial["permutations"][:1]
        partial["permutation_timings"] = partial["permutation_timings"][:1]
        partial["error"] = "KeyboardInterrupt: "
        checkpoint.write_text(json.dumps(partial), encoding="utf-8")
        with patch.object(ws, "nested_cv", wraps=ws.nested_cv) as nested:
            result = self.run_permutations(output_dir=None, resume_from=checkpoint)
            self.assertEqual(nested.call_count, 1)
        self.assertEqual(result["permutations"], full["permutations"])
        self.assertEqual(result["p_value"], full["p_value"])
        self.assertFalse(checkpoint.with_suffix(".lock").exists())
        before = checkpoint.read_bytes()
        with patch.object(ws, "nested_cv") as nested:
            self.run_permutations(output_dir=None, resume_from=checkpoint)
            nested.assert_not_called()
        self.assertEqual(checkpoint.read_bytes(), before)
        for corruption in ("seed", "labels", "scores", "identity"):
            broken = json.loads(json.dumps(partial))
            if corruption == "seed":
                broken["random_seed"] = 7
            elif corruption == "labels":
                broken["permutations"][0]["permutation_indices"].reverse()
            elif corruption == "scores":
                broken["permutations"][0]["accuracy"] = -1
            else:
                broken["input_sha256"] = "wrong"
            checkpoint.write_text(json.dumps(broken), encoding="utf-8")
            before = checkpoint.read_bytes()
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                self.run_permutations(output_dir=None, resume_from=checkpoint)
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_resume_lock_prevents_concurrent_run(self):
        directory = self.root / "locked"
        directory.mkdir()
        checkpoint = directory / "permutation_results.json"
        checkpoint.with_suffix(".lock").write_text("other process")
        with self.assertRaises(FileExistsError), patch.object(ws, "nested_cv") as nested:
            self.run_permutations(output_dir=None, resume_from=checkpoint)
            nested.assert_not_called()

    def test_p_value_ties_plus_one_and_validation(self):
        self.assertEqual(permutation.empirical_p_value(.75, [.5, .75, .8]), (.75, 2))
        self.assertEqual(permutation.empirical_p_value(.9, [.1, .2, .3]), (.25, 0))
        self.assertEqual(permutation.empirical_p_value(.5, [.5, .6, .7]), (1.0, 3))
        self.assertEqual(permutation.empirical_p_value(.5, [.5 - 1e-16]), (1.0, 1))
        for values in ([], [np.nan], [1.1], [[.5]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                permutation.empirical_p_value(.5, values)

    def test_failure_keeps_completed_only_and_no_final_p_value(self):
        nested = ws.nested_cv
        calls = []

        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("synthetic interruption")
            return nested(*args, **kwargs)

        with patch.object(ws, "nested_cv", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                self.run_permutations()
        directory = self.root / "permutations"
        result = json.loads((directory / "permutation_results.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["n_completed"], 1)
        self.assertEqual(len(result["permutations"]), 1)
        self.assertIsNone(result["p_value"])
        self.assertFalse((directory / "permutation_summary.mat").exists())

    def test_no_overwrite_and_bad_counts_rejected_before_fitting(self):
        with patch.object(ws, "nested_cv") as nested:
            for value in (0, -1, True, 1.5):
                with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                    self.run_permutations(n_permutations=value)
            with self.assertRaises(ValueError):
                self.run_permutations(random_seed=-1)
            with self.assertRaises(FileExistsError):
                self.run_permutations(output_dir=self.root)
            nested.assert_not_called()

    def test_modified_input_hash_is_rejected(self):
        with h5py.File(self.path, "r+") as f:
            f["clean_eeg"][0, 0, 0] += .1
        with patch.object(ws, "nested_cv") as nested:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.run_permutations()
            nested.assert_not_called()

    def test_corrupt_observed_score_grid_and_indices_rejected(self):
        original = self.mat_path.read_bytes()
        for corruption in ("score", "grid", "indices", "runtime", "folds"):
            self.mat_path.write_bytes(original)
            stored = loadmat(self.mat_path, simplify_cells=True)
            details = json.loads(stored["nested_cv_details_json"])
            if corruption == "score":
                stored["accuracy"] = -1.0
            elif corruption == "grid":
                details["param_grid"]["svm__C"] = [999.0]
            elif corruption == "indices":
                details["folds"][0]["test_indices"] = [0]
            elif corruption == "runtime":
                stored["config"]["runtime_versions_json"] = "{}"
            else:
                stored["config"]["inner_k_fold"] = 2
            stored["nested_cv_details_json"] = json.dumps(details)
            savemat(self.mat_path, {k: v for k, v in stored.items() if not k.startswith("__")})
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                permutation.load_observed_result(self.mat_path)

    def test_legacy_report_fallback_and_validate_only(self):
        stored = loadmat(self.mat_path, simplify_cells=True)
        for key in ("input_sha256", "cv_protocol", "runtime_versions_json"):
            stored["config"].pop(key)
        savemat(self.mat_path, {k: v for k, v in stored.items() if not k.startswith("__")})
        report_path = self.root / "report.json"
        report = {"run_id": "synthetic-nested-cv", "ws_output": str(self.path),
                  "ws_sha256": ws.file_sha256(self.path)}
        report_path.write_text(json.dumps(report))
        with self.assertWarns(RuntimeWarning), patch.object(ws, "nested_cv") as nested, redirect_stdout(StringIO()):
            result = permutation.main(["--observed-mat", str(self.mat_path),
                                       "--stage4-report", str(report_path), "--validate-only"])
            self.assertEqual(result["provenance"], "legacy_stage4_report")
            nested.assert_not_called()
        report["ws_sha256"] = "wrong"
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "report/hash mismatch"):
            permutation.load_observed_result(self.mat_path, stage4_report_path=report_path)


# 不让 unittest 将复用的数据夹具类误认为本模块的测试类。
del _Fixture


if __name__ == "__main__":
    unittest.main()