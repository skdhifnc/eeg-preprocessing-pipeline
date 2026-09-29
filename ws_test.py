"""嵌套 CSP 分类回归测试：只用合成 EEG 和临时文件，不修改正式结果。"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import mne
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import ParameterGrid, StratifiedKFold

import main as classification
import ws


class NestedCVTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "clean.hdf5"
        self.labels = np.tile([1, 2], 10).astype("int32")
        self.data = np.random.default_rng(42).normal(size=(20, 4, 48))
        self.data[self.labels == 1, 0] *= 1.7
        self.data[self.labels == 2, 1] *= 1.7
        self.grid = {"csp__reg": [0.1], "csp__n_components": [2],
                     "csp__log": [True], "svm__C": [1.0]}
        self.write_data(self.data, self.labels)

    def write_data(self, data, labels):
        with h5py.File(self.path, "w") as f:
            f.create_dataset("clean_eeg", data=data)
            f.create_dataset("labels", data=labels)
            f.create_dataset("channel_names", data=[f"Ch{i}" for i in range(data.shape[1])],
                             dtype=h5py.string_dtype())
            times = np.arange(data.shape[2], dtype=float) / 250
            f.create_dataset("times", data=times)
            f.create_dataset("event_samples", data=np.arange(len(labels)) * 100)
            f.attrs.update(stage=4, sampling_rate=250.0, unit="microvolts",
                           axis_order="epochs,channels,times", run_id="synthetic-nested-cv",
                           label_map=json.dumps({"left": 1, "right": 2}),
                           tmin=float(times[0]), tmax=float(times[-1]))

    def run_cv(self, **kwargs):
        kwargs.setdefault("param_grid", self.grid)
        with mne.utils.use_log_level("ERROR"), redirect_stdout(StringIO()):
            return ws.train_test(self.path, **kwargs)

    def test_default_grid_and_low_channel_grid(self):
        grid = ws._search_grid(None, 32)
        self.assertEqual(len(ParameterGrid(grid)), 72)
        self.assertEqual(grid["csp__reg"], [0.05, 0.1, 0.2, 0.4, "ledoit_wolf", "oas"])
        self.assertEqual(grid["csp__log"], [False, True])
        self.assertEqual(ws._search_grid(None, 2)["csp__n_components"], [2])
        grid["csp__reg"].clear()
        self.assertEqual(len(ws.DEFAULT_PARAM_GRID["csp__reg"]), 6)

    def test_real_default_ten_by_five_and_legacy_return(self):
        average, scores, details = self.run_cv(return_details=True)
        self.assertEqual(len(scores), 10)
        self.assertEqual(details["inner_k_fold"], 5)
        self.assertEqual(average, float(np.mean(scores)))
        self.assertTrue(all(0 <= value <= 1 for value in scores))
        tests = [i for fold in details["folds"] for i in fold["test_indices"]]
        self.assertEqual(sorted(tests), list(range(20)))
        self.assertEqual(json.loads(json.dumps(details)), details)
        # 确认默认二元返回与详情返回一致，固定划分可复现。
        repeated = self.run_cv()
        self.assertEqual(len(repeated), 2)
        self.assertEqual(repeated, (average, scores))

    def test_every_candidate_inner_fold_and_outer_refit_are_isolated(self):
        grid = dict(self.grid, svm__C=[0.1, 1.0])
        fit_events, scaler_events = [], []
        lookup = {float(epoch[0, 0]): index for index, epoch in enumerate(self.data)}
        original_fit = ws.CSP.fit
        original_scaler_fit = ws.StandardScaler.fit

        def tracked_fit(estimator, X, y):
            indices = tuple(lookup[float(epoch[0, 0])] for epoch in X)
            fit_events.append((estimator, indices))
            return original_fit(estimator, X, y)

        def tracked_scaler_fit(estimator, X, y=None, sample_weight=None):
            scaler_events.append((estimator, len(X)))
            return original_scaler_fit(estimator, X, y, sample_weight=sample_weight)

        with patch.object(ws.CSP, "fit", tracked_fit), \
                patch.object(ws.StandardScaler, "fit", tracked_scaler_fit):
            _, scores, details = self.run_cv(param_grid=grid, return_details=True)

        # 每外层折：2 个候选各自 5 次拟合 + 1 次最佳参数 refit，共 110 次。
        self.assertEqual(len(fit_events), 110)
        self.assertEqual(len(scaler_events), 110)
        self.assertEqual(len({id(event[0]) for event in fit_events}), 110)
        self.assertEqual(len({id(event[0]) for event in scaler_events}), 110)
        outer = StratifiedKFold(10, shuffle=False)
        for fold_index, (train, test) in enumerate(outer.split(self.data, self.labels)):
            inner = list(StratifiedKFold(5, shuffle=False).split(self.data[train], self.labels[train]))
            expected = [tuple(train[inner_train]) for inner_train, _ in inner] * 2 + [tuple(train)]
            block = fit_events[fold_index * 11:(fold_index + 1) * 11]
            self.assertEqual([event[1] for event in block], expected)
            self.assertEqual([event[1] for event in scaler_events[fold_index * 11:(fold_index + 1) * 11]],
                             [len(indices) for indices in expected])
            for _, indices in block:
                self.assertFalse(set(indices) & set(test))
            fold = details["folds"][fold_index]
            self.assertEqual(fold["train_indices"], train.tolist())
            self.assertEqual(fold["test_indices"], test.tolist())
            self.assertEqual(fold["outer_score"], scores[fold_index])
            self.assertIn(fold["best_params"]["svm__C"], [0.1, 1.0])

    def test_full_grid_all_shrinkage_log_and_component_options_fit(self):
        # 小型合成数据实际跑完整默认 72 组、10x5 嵌套搜索（3610 次 fit）。
        average, scores, details = self.run_cv(param_grid=None, return_details=True)
        self.assertEqual(details["n_candidates"], 72)
        self.assertEqual(len(scores), 10)
        self.assertTrue(np.isfinite(average))
        for fold in details["folds"]:
            self.assertTrue(np.isfinite(fold["best_inner_score"]))
            for key, value in fold["best_params"].items():
                self.assertIn(value, ws.DEFAULT_PARAM_GRID[key])

    def test_invalid_fold_types_and_counts_fail_before_loading(self):
        with patch.object(ws, "load_clean_data") as load:
            for name in ("k_fold", "inner_k_fold"):
                for value in (True, 2.5, "5"):
                    with self.subTest(name=name, value=value), self.assertRaises(TypeError):
                        self.run_cv(**{name: value})
                for value in (0, 1):
                    with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                        self.run_cv(**{name: value})
            load.assert_not_called()

    def test_insufficient_outer_or_inner_samples_fail_before_search(self):
        with patch.object(ws, "GridSearchCV") as search:
            with self.assertRaisesRegex(ValueError, "k_fold=11"):
                self.run_cv(k_fold=11)
            with self.assertRaisesRegex(ValueError, "Outer fold 1: inner_k_fold=10"):
                self.run_cv(inner_k_fold=10)
            search.assert_not_called()

    def test_invalid_grids_fail_before_search(self):
        invalid = [{}, {"svm__kernel": ["rbf"]}, {"csp__reg": []},
                   {"csp__reg": ["wrong"]}, {"csp__reg": [-0.1]},
                   {"csp__n_components": [5]}, {"csp__n_components": [True]},
                   {"csp__log": [1]}, {"svm__C": [0]}, {"svm__C": [np.inf]}]
        with patch.object(ws, "GridSearchCV") as search:
            for grid in invalid:
                with self.subTest(grid=grid), self.assertRaises(ValueError):
                    self.run_cv(param_grid=grid)
            search.assert_not_called()

    def test_numpy_candidates_are_serializable(self):
        grid = ws._search_grid({"csp__n_components": [np.int64(2)],
                                "csp__log": [np.bool_(True)], "svm__C": [np.float64(1)]}, 4)
        self.assertEqual(json.loads(json.dumps(grid)), grid)

    def test_cache_equivalence_and_parallel_worker_limits(self):
        grid = dict(ws.DEFAULT_PARAM_GRID)
        with mne.utils.use_log_level("ERROR"), redirect_stdout(StringIO()):
            baseline = ws.nested_cv(self.data, self.labels, k_fold=2, inner_k_fold=2,
                                    param_grid=grid, return_details=True, cache_transformers=False)
            cached = ws.nested_cv(self.data, self.labels, k_fold=2, inner_k_fold=2,
                                  param_grid=grid, return_details=True, cache_transformers=True)
            parallel = ws.nested_cv(self.data, self.labels, k_fold=2, inner_k_fold=2,
                                    param_grid=self.grid, return_details=True,
                                    cache_transformers=True, n_jobs=2, native_threads=1)
            serial = ws.nested_cv(self.data, self.labels, k_fold=2, inner_k_fold=2,
                                  param_grid=self.grid, return_details=True,
                                  cache_transformers=True, n_jobs=1, native_threads=1)
        self.assertEqual(baseline, cached)
        self.assertEqual(parallel[:2], serial[:2])
        self.assertEqual(parallel[2]["folds"], serial[2]["folds"])

    def test_fit_errors_are_not_silently_ignored(self):
        with patch.object(ws.CSP, "fit", side_effect=RuntimeError("synthetic fit failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic fit failure"):
                self.run_cv()

    def test_main_writes_nested_metadata_without_touching_input(self):
        before = self.path.read_bytes()
        mat_path, txt_path = self.root / "result.mat", self.root / "result.txt"
        with mne.utils.use_log_level("ERROR"), redirect_stdout(StringIO()):
            average, scores = classification.run_classification(
                clean_data_path=self.path, result_mat_path=mat_path,
                result_txt_path=txt_path, param_grid=self.grid,
            )
        stored = loadmat(mat_path, simplify_cells=True)
        self.assertEqual(stored["accuracy"], average)
        np.testing.assert_array_equal(stored["fold_scores"], scores)
        details = json.loads(stored["nested_cv_details_json"])
        self.assertEqual(len(details["folds"]), 10)
        self.assertEqual(stored["config"]["inner_k_fold"], 5)
        np.testing.assert_array_equal(stored["best_inner_scores"],
                                      [fold["best_inner_score"] for fold in details["folds"]])
        text = txt_path.read_text(encoding="utf-8")
        self.assertIn("Inner folds: 5", text)
        self.assertEqual(text.count("best params="), 10)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()