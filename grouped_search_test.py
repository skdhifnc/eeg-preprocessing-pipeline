"""分组搜索逐候选与 GridSearchCV 对照，包含训练隔离、并列规则和低秩数据。"""
from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np
from mne.utils import use_log_level
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

import ws


class GroupedSearchTest(unittest.TestCase):
    def setUp(self):
        self.data = np.random.default_rng(19).normal(size=(23, 4, 64))
        self.labels = np.arange(23) % 2

    def compare(self, X, y, grid, n_jobs=1):
        with threadpool_limits(limits=2), use_log_level("ERROR"), redirect_stdout(StringIO()):
            reference = GridSearchCV(ws._pipeline(), grid, cv=StratifiedKFold(3),
                                     scoring="accuracy", error_score="raise").fit(X, y)
            grouped = ws.grouped_search(X, y, grid, inner_k_fold=3, n_jobs=n_jobs)
        expected = np.array([reference.cv_results_[f"split{i}_test_score"] for i in range(3)]).T
        self.assertEqual(grouped["params"], reference.cv_results_["params"])
        np.testing.assert_array_equal(grouped["split_scores"], expected)
        np.testing.assert_array_equal(grouped["mean_scores"], reference.cv_results_["mean_test_score"])
        self.assertEqual(grouped["best_index"], reference.best_index_)
        np.testing.assert_array_equal(grouped["best_estimator"].predict(X), reference.predict(X))
        np.testing.assert_allclose(grouped["best_estimator"].decision_function(X),
                                   reference.decision_function(X), atol=1e-12, rtol=1e-12)

    def test_full_grid_and_multiple_permutations(self):
        for seed in (3, 7):
            with self.subTest(seed=seed):
                y = self.labels[np.random.default_rng(seed).permutation(len(self.labels))]
                self.compare(self.data, y, ws.DEFAULT_PARAM_GRID)

    def test_defaults_duplicate_candidates_ties_and_parallel(self):
        # 未列入网格的参数使用原 Pipeline 的固定默认值；重复候选强制产生并列。
        grid = {"svm__C": [1., 1., .1]}
        self.compare(self.data, self.labels, grid, n_jobs=2)

    def test_rank_deficient_input_and_non_sorted_component_candidates(self):
        data = self.data.copy()
        data[:, 3] = data[:, 0] + data[:, 1]
        self.compare(data, self.labels, {"csp__reg": [.1, "oas", "ledoit_wolf"],
                                        "csp__log": [True, False], "csp__n_components": [4, 2],
                                        "svm__C": [.1, 1.]})

    def test_fits_use_only_inner_training_samples_and_no_mutation(self):
        X, y = self.data.copy(), self.labels.copy()
        fits = []
        original = ws.CSP.fit

        def tracked(estimator, data, labels):
            fits.append((estimator, data.copy(), labels.copy()))
            return original(estimator, data, labels)

        with patch.object(ws.CSP, "fit", tracked), use_log_level("ERROR"):
            ws.grouped_search(X, y, ws.DEFAULT_PARAM_GRID, inner_k_fold=3)
        n_regs = len(ws.DEFAULT_PARAM_GRID["csp__reg"])
        self.assertEqual(len(fits), 3 * n_regs + 1)
        self.assertEqual(len({id(f[0]) for f in fits}), len(fits))
        for fold, (train, validation) in enumerate(StratifiedKFold(3).split(X, y)):
            for _, received_X, received_y in fits[fold * n_regs:(fold + 1) * n_regs]:
                np.testing.assert_array_equal(received_X, X[train])
                np.testing.assert_array_equal(received_y, y[train])
        np.testing.assert_array_equal(fits[-1][1], X)
        np.testing.assert_array_equal(X, self.data)
        np.testing.assert_array_equal(y, self.labels)

    def test_nested_outer_scores_and_params_match_reference(self):
        grid = {"csp__reg": [.1, "oas"], "csp__log": [True, False],
                "csp__n_components": [2, 4], "svm__C": [.1, 1.]}
        with use_log_level("ERROR"), redirect_stdout(StringIO()):
            reference = ws.nested_cv(self.data, self.labels, k_fold=3, inner_k_fold=3,
                                     param_grid=grid, return_details=True)
            grouped = ws.nested_cv(self.data, self.labels, k_fold=3, inner_k_fold=3,
                                   param_grid=grid, return_details=True, search_engine="grouped")
        self.assertEqual(reference, grouped)

    def test_full_ten_by_five_grid_has_expected_csp_fits(self):
        # 每类至少10个epoch，10x5合法；真正执行完整默认候选，不用mock替换算法。
        original = ws.CSP.fit
        count = []

        def tracked(estimator, X, y):
            count.append(1)
            return original(estimator, X, y)

        with patch.object(ws.CSP, "fit", tracked), use_log_level("ERROR"), redirect_stdout(StringIO()):
            result = ws.nested_cv(self.data, self.labels, search_engine="grouped")
        self.assertEqual(len(count), 10 * (5 * len(ws.DEFAULT_PARAM_GRID["csp__reg"]) + 1))
        self.assertEqual(len(result[1]), 10)


if __name__ == "__main__":
    unittest.main()