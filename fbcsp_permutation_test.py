"""FBCSP 标签置换/断点续跑测试，全部使用临时合成数据。"""
import json
import unittest
from unittest.mock import patch

import numpy as np

import fbcsp
import fbcsp_main
import fbcsp_permutation as permutation
import fbcsp_test as fixtures


class PermutationTest(fixtures.Fixture):
    def setUp(self):
        super().setUp()
        report, _ = self.publication()
        self.grid["svm__C"] = [1.]
        fbcsp_main.run_classification(report_path=report, output_dir=self.root / "observed",
                                      k_fold=2, inner_k_fold=2, param_grid=self.grid,
                                      include_baseline=False)
        self.observed = self.root / "observed" / "classification_result.json"

    def test_full_refits_seed_resume_and_corruption(self):
        calls = []
        original = fbcsp._NativeFloat64CSP.fit

        def tracked(estimator, X, y):
            calls.append((X.copy(), y.copy()))
            return original(estimator, X, y)

        with patch.object(fbcsp._NativeFloat64CSP, "fit", tracked):
            result = permutation.run_permutation_test(self.observed, n_permutations=2,
                                                      output_dir=self.root / "permutations")
        self.assertEqual(len(calls), 2 * 2 * (2 + 1) * 2)
        rng = np.random.default_rng(42)
        for item in result["permutations"]:
            indices = rng.permutation(len(self.y))
            self.assertEqual(item["permutation_indices"], indices.tolist())
            self.assertEqual(item["permuted_labels"], self.y[indices].tolist())
        for i in range(0, len(calls), 2):
            np.testing.assert_array_equal(calls[i][1], calls[i + 1][1])
        checkpoint = self.root / "permutations" / "permutation_results.json"
        before = checkpoint.read_bytes()
        completed = permutation.run_permutation_test(self.observed, n_permutations=2,
                                                     resume_from=checkpoint)
        self.assertEqual(completed, result)
        self.assertEqual(checkpoint.read_bytes(), before)
        partial = json.loads(before)
        partial.update(status="failed", n_completed=1, p_value=None, n_extreme=None)
        partial["permutations"] = partial["permutations"][:1]
        checkpoint.write_text(json.dumps(partial))
        resumed = permutation.run_permutation_test(self.observed, n_permutations=2,
                                                   resume_from=checkpoint)
        self.assertEqual(resumed, result)
        partial["permutations"][0]["permutation_indices"].reverse()
        checkpoint.write_text(json.dumps(partial))
        with self.assertRaisesRegex(ValueError, "sequence"):
            permutation.run_permutation_test(self.observed, n_permutations=2, resume_from=checkpoint)
        self.assertFalse(checkpoint.with_suffix(".lock").exists())

    def test_observed_validation_and_p_value(self):
        self.assertEqual(permutation.empirical_p_value(.75, [.5, .75, .8]), (.75, 2))
        for alteration in ("accuracy", "runtime_versions", "input"):
            original = self.observed.read_bytes()
            record = json.loads(original)
            record[alteration] = -1 if alteration == "accuracy" else {}
            self.observed.write_text(json.dumps(record))
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                permutation.load_observed(self.observed)
            self.observed.write_bytes(original)

    def test_old_protocol_is_rejected_even_with_same_grid(self):
        record = json.loads(self.observed.read_bytes())
        record["cv_details"]["cv_protocol"] = "nested_fbcsp_v1"
        self.observed.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "protocol"):
            permutation.load_observed(self.observed)

    def test_failure_checkpoint_and_no_overwrite(self):
        original = fbcsp.nested_cv
        count = 0

        def interrupted(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("synthetic interruption")
            return original(*args, **kwargs)

        directory = self.root / "failed"
        with patch.object(fbcsp, "nested_cv", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                permutation.run_permutation_test(self.observed, output_dir=directory, n_permutations=2)
        checkpoint = directory / "permutation_results.json"
        record = json.loads(checkpoint.read_text())
        self.assertEqual(record["n_completed"], 1)
        self.assertEqual(record["status"], "failed")
        self.assertIsNone(record["p_value"])
        with self.assertRaises(FileExistsError):
            permutation.run_permutation_test(self.observed, output_dir=directory)


if __name__ == "__main__":
    unittest.main()