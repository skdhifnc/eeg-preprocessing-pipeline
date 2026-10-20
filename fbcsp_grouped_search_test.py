"""FBCSP 分组搜索：真实 MNE/SVM 数值对照、逐带切片、log 与训练隔离。"""
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
from mne.utils import use_log_level
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import fbcsp
import ws


class GroupedSearchTest(unittest.TestCase):
    def setUp(self):
        # 不等大的折、五带、不同带/通道尺度；避免错误拼接恰好产生相同结果。
        self.X = np.random.default_rng(81).normal(size=(23, 5, 4, 48))
        self.X *= np.arange(1, 6)[None, :, None, None]
        self.y = (np.arange(len(self.X)) % 2 + 1).astype(np.int32)
        for context in (threadpool_limits(limits=2), use_log_level("ERROR"),
                        redirect_stdout(StringIO())):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def compare(self, X, y, grid, n_jobs=1):
        reference = GridSearchCV(fbcsp._pipeline(), grid, cv=StratifiedKFold(3),
                                 scoring="accuracy", error_score="raise").fit(X, y)
        grouped = fbcsp.grouped_search(X, y, grid, inner_k_fold=3, n_jobs=n_jobs)
        expected = np.array([reference.cv_results_[f"split{i}_test_score"] for i in range(3)]).T
        self.assertEqual(grouped["params"], reference.cv_results_["params"])
        np.testing.assert_array_equal(grouped["split_scores"], expected)
        np.testing.assert_array_equal(grouped["mean_scores"], reference.cv_results_["mean_test_score"])
        self.assertEqual(grouped["best_index"], reference.best_index_)
        self.assertEqual(grouped["params"][grouped["best_index"]], reference.best_params_)
        np.testing.assert_array_equal(grouped["best_estimator"].predict(X), reference.predict(X))
        np.testing.assert_allclose(grouped["best_estimator"].decision_function(X),
                                   reference.decision_function(X), rtol=1e-12, atol=1e-12)
        self.assertIsNone(grouped["best_estimator"].memory)

    def test_current_grid_and_permuted_labels(self):
        for seed in (3, 7):
            with self.subTest(seed=seed):
                y = self.y[np.random.default_rng(seed).permutation(len(self.y))]
                self.compare(self.X, y, fbcsp.FBCSP_PARAM_GRID)

    def test_both_logs_rank_deficiency_unsorted_counts(self):
        X = self.X.copy()
        X[:, :, 3] = X[:, :, 0] + X[:, :, 1]
        self.compare(X, self.y, {"fbcsp__reg": [.1, "oas", "ledoit_wolf"],
                                 "fbcsp__log": [False, True],
                                 "fbcsp__n_components": [4, 2], "svm__C": [.1, 1.]})

    def test_partial_grid_ties_and_parallel(self):
        self.compare(self.X, self.y, {"svm__C": [1., 1., .1]}, n_jobs=2)

    def test_features_match_mne_before_scaler_and_svm(self):
        # 故意选择不连续、非排序的外层训练集，并用局部索引建立内层切片。
        outer = np.array([19, 2, 9, 0, 17, 6, 11, 4, 15, 8, 21, 10, 13, 12, 1, 14])
        X, y = self.X[outer], self.y[outer]
        train, validation = next(StratifiedKFold(2).split(X, y))
        before = X.copy()
        grid = {"fbcsp__reg": [.1, "ledoit_wolf"], "fbcsp__n_components": [4, 2],
                "fbcsp__log": [False, True], "svm__C": [.1, 1.]}
        candidates = list(ParameterGrid(grid))
        raw_features, svm_inputs, score_inputs, scaler_inputs, fits = [], [], [], [], []
        original_features = fbcsp._grouped_features
        original_fit = fbcsp._NativeFloat64CSP.fit
        original_svm_fit = fbcsp.SVC.fit
        original_svm_score = fbcsp.SVC.score
        original_scaler_fit = StandardScaler.fit

        def features(csps, train_data, val_data, count, log):
            actual = original_features(csps, train_data, val_data, count, log)
            raw_features.append((csps[0].reg, count, log, actual))
            return actual

        def fit(estimator, data, labels):
            self.assertEqual(estimator.cov_est, "concat")
            fits.append((data.copy(), labels.copy()))
            return original_fit(estimator, data, labels)

        def svm_fit(estimator, data, labels, *args, **kwargs):
            svm_inputs.append((estimator.C, data.copy(), labels.copy()))
            return original_svm_fit(estimator, data, labels, *args, **kwargs)

        def scaler_fit(estimator, data, *args, **kwargs):
            scaler_inputs.append(data.copy())
            return original_scaler_fit(estimator, data, *args, **kwargs)

        def svm_score(estimator, data, labels, *args, **kwargs):
            score_inputs.append((data.copy(), labels.copy()))
            return original_svm_score(estimator, data, labels, *args, **kwargs)

        with patch.object(fbcsp, "_grouped_features", features), \
                patch.object(fbcsp._NativeFloat64CSP, "fit", fit), \
                patch.object(fbcsp.SVC, "fit", svm_fit), \
                patch.object(fbcsp.SVC, "score", svm_score), \
                patch.object(StandardScaler, "fit", scaler_fit):
            scores = fbcsp._grouped_inner_fold(X, y, train, validation, candidates, 2)
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(len(fits), 2 * 5)  # reg × band，与 count/log/C 无关
        for reg_index in range(2):
            for band in range(5):
                received_X, received_y = fits[reg_index * 5 + band]
                np.testing.assert_array_equal(received_X, self.X[outer[train], band])
                np.testing.assert_array_equal(received_y, self.y[outer[train]])
        self.assertEqual(len(raw_features), 2 * 2 * 2)
        self.assertEqual(len(svm_inputs), len(candidates))
        self.assertEqual(len(score_inputs), len(candidates))
        for group, (reg, count, log, (actual_train, actual_val)) in enumerate(raw_features):
            reference = fbcsp.FBCSPTransformer(reg=reg, n_components=count, log=log).fit(X[train], y[train])
            expected_train = reference.transform(X[train])
            expected_val = reference.transform(X[validation])
            self.assertEqual(actual_train.shape, (len(train), 5 * count))
            np.testing.assert_allclose(actual_train, expected_train, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual_val, expected_val, rtol=1e-12, atol=1e-12)
            # 检查逐带块，而不是误用拼接后的前 5*count 列。
            for band in range(5):
                np.testing.assert_allclose(actual_val[:, band * count:(band + 1) * count],
                                           reference.band_csps_[band].transform(X[validation, band]),
                                           rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(scaler_inputs[group], actual_train)
            scaler = StandardScaler().fit(expected_train)
            for offset, C in enumerate(grid["svm__C"]):
                got_C, svm_X, svm_y = svm_inputs[group * 2 + offset]
                self.assertEqual(got_C, C)
                np.testing.assert_allclose(svm_X, scaler.transform(expected_train), rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(svm_y, y[train])
                val_X, val_y = score_inputs[group * 2 + offset]
                np.testing.assert_allclose(val_X, scaler.transform(expected_val), rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(val_y, y[validation])
        np.testing.assert_array_equal(X, before)

    def test_nested_training_isolation_and_outer_refit(self):
        fits = []
        original = fbcsp._NativeFloat64CSP.fit

        def fit(estimator, X, y):
            fits.append((X.copy(), y.copy()))
            return original(estimator, X, y)

        with patch.object(fbcsp._NativeFloat64CSP, "fit", fit):
            fbcsp.nested_cv(self.X, self.y, k_fold=2, inner_k_fold=2)
        expected = []
        for outer_train, _ in StratifiedKFold(2).split(self.X, self.y):
            for inner_train, _ in StratifiedKFold(2).split(self.X[outer_train], self.y[outer_train]):
                for _ in ws.DEFAULT_PARAM_GRID["csp__reg"]:
                    expected.extend((outer_train[inner_train], band) for band in range(5))
            expected.extend((outer_train, band) for band in range(5))
        self.assertEqual(len(fits), len(expected))
        for (received_X, received_y), (indices, band) in zip(fits, expected):
            np.testing.assert_array_equal(received_X, self.X[indices, band])
            np.testing.assert_array_equal(received_y, self.y[indices])

    def test_nested_reference_equivalence_and_baseline(self):
        grid = {"fbcsp__reg": [.1, "oas"], "fbcsp__n_components": [4, 2],
                "fbcsp__log": [True, False], "svm__C": [.1, 1.]}
        for baseline in (False, True):
            X = self.X[:, :1] if baseline else self.X
            options = dict(k_fold=2, inner_k_fold=2, param_grid=grid,
                           return_details=True, baseline=baseline)
            reference = fbcsp.nested_cv(X, self.y, search_engine="reference", **options)
            grouped = fbcsp.nested_cv(X, self.y, **options)
            self.assertEqual(reference, grouped)

    def test_complete_default_ten_by_five_fits_550_band_csps(self):
        original = fbcsp._NativeFloat64CSP.fit
        fits = []

        def fit(estimator, X, y):
            fits.append((estimator.reg, len(X)))
            return original(estimator, X, y)

        with patch.object(fbcsp._NativeFloat64CSP, "fit", fit):
            _, scores, details = fbcsp.nested_cv(self.X, self.y, return_details=True)
        self.assertEqual(len(fits), 550)
        self.assertEqual(len(scores), 10)
        self.assertEqual(details["n_candidates"], 12)
        self.assertEqual(details["cv_protocol"], "nested_fbcsp_v2_grouped_concat")
        self.assertTrue(all(len(fold["feature_metadata"]) in (10, 20) for fold in details["folds"]))


if __name__ == "__main__":
    unittest.main()