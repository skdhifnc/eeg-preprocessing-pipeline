"""FBCSP 合成数据测试；不访问真实 EEG，不改写正式输出。"""
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
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import fbcsp
import fbcsp_main
import fbcsp_stage4
import ws
import EEG_processing as stage4


class Fixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.X = np.random.default_rng(24).normal(size=(20, 2, 4, 64))
        self.y = np.tile([1, 2], 10).astype(np.int32)
        self.X[self.y == 1, :, 0] *= 1.7
        self.X[self.y == 2, :, 1] *= 1.7
        self.grid = {"fbcsp__reg": [0.1], "fbcsp__n_components": [2],
                     "fbcsp__log": [True], "svm__C": [0.1, 1.]}
        self._enter(mne.utils.use_log_level("ERROR"))
        self._enter(threadpool_limits(limits=1))
        self._enter(redirect_stdout(StringIO()))

    def _enter(self, context):
        value = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        return value

    def publication(self):
        paths = [self.root / name for name in ("full.hdf5", "fb.hdf5", "base.hdf5")]
        times = np.arange(self.X.shape[-1]) / 250.
        samples = np.arange(len(self.y), dtype=np.int64) * 1000
        names = np.array([f"Ch{i}" for i in range(4)], dtype="S")
        bands = ((8., 12.), (12., 16.))
        metadata = dict(stage=4, run_id="synthetic", source_raw_sha256="a" * 64,
                        sampling_rate=250., unit="microvolts", label_map={"left": 1, "right": 2},
                        axis_order="epochs,bands,channels,times")
        bundle = (self.X.astype(np.float32), self.y, names, times, samples, bands)
        for path in paths[:2]:
            fbcsp_stage4._write_hdf5(path, bundle, metadata, "synthetic")
        base_metadata = dict(metadata, axis_order="epochs,channels,times", tmin=times[0],
                             tmax=times[-1], filter_l_freq=8., filter_h_freq=30.)
        stage4.write_hdf5(paths[2], self.X[:, 0].astype(np.float32), self.y, names, times,
                          samples, base_metadata)
        report = dict(metadata, bands=[list(b) for b in bands], reference_filter_band=[8., 30.])
        for kind, path in zip(("full", "ws", "baseline"), paths):
            report[f"{kind}_output"] = str(path)
            report[f"{kind}_sha256"] = fbcsp.file_sha256(path)
        report_path = self.root / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path, paths


class TransformerTest(Fixture):
    def test_grid_matches_baseline_and_clone(self):
        self.assertEqual(fbcsp.FBCSP_PARAM_GRID,
                         {k.replace("csp__", "fbcsp__"): v for k, v in ws.DEFAULT_PARAM_GRID.items()})
        self.assertEqual(len(ParameterGrid(fbcsp._validate_grid(None))), 12)
        self.assertIsInstance(clone(fbcsp.FBCSPTransformer()), fbcsp.FBCSPTransformer)

    def test_independent_bands_match_mne_and_no_mutation(self):
        before = self.X.copy()
        for count in (2, 4):
            for log in (False, True):
                model = fbcsp.FBCSPTransformer(reg=.1, n_components=count, log=log).fit(self.X, self.y)
                transformed = model.transform(self.X)
                self.assertEqual(transformed.shape, (20, 2 * count))
                self.assertEqual(len(set(map(id, model.band_csps_))), 2)
                for band in range(2):
                    reference = ws._NativeFloat64CSP(n_components=count, reg=.1, log=log,
                                                      norm_trace=False).fit(self.X[:, band], self.y)
                    np.testing.assert_allclose(transformed[:, band * count:(band + 1) * count],
                                               reference.transform(self.X[:, band]), atol=1e-12)
        np.testing.assert_array_equal(self.X, before)

    def test_nested_fits_are_only_inner_training_and_outer_refit(self):
        fits, scalers = [], []
        original, original_scaler = fbcsp._NativeFloat64CSP.fit, StandardScaler.fit

        def tracked(estimator, X, y):
            fits.append((X.copy(), y.copy()))
            return original(estimator, X, y)

        def tracked_scaler(estimator, X, y=None, **kwargs):
            scalers.append(len(X))
            return original_scaler(estimator, X, y, **kwargs)

        one = dict(self.grid, svm__C=[1.])
        with patch.object(fbcsp._NativeFloat64CSP, "fit", tracked), \
                patch.object(StandardScaler, "fit", tracked_scaler):
            average, scores, details = fbcsp.nested_cv(self.X, self.y, k_fold=2, inner_k_fold=2,
                                                      param_grid=one, return_details=True,
                                                      cache_transformers=False)
        expected = []
        for train, test in StratifiedKFold(2).split(self.X, self.y):
            for inner_train, _ in StratifiedKFold(2).split(self.X[train], self.y[train]):
                expected.append(train[inner_train])
            expected.append(train)
        self.assertEqual(len(fits), len(expected) * 2)
        for index, indices in enumerate(expected):
            for band in range(2):
                np.testing.assert_array_equal(fits[2 * index + band][0], self.X[indices, band])
                np.testing.assert_array_equal(fits[2 * index + band][1], self.y[indices])
        self.assertEqual(scalers, [len(indices) for indices in expected])
        self.assertEqual(average, np.mean(scores))
        self.assertEqual(sorted(i for fold in details["folds"] for i in fold["test_indices"]), list(range(20)))

    def test_legacy_cache_flag_and_parallel_equivalence_and_paired_baseline(self):
        options = dict(k_fold=2, inner_k_fold=2, param_grid=self.grid, return_details=True)
        no_cache = fbcsp.nested_cv(self.X, self.y, cache_transformers=False, **options)
        cached = fbcsp.nested_cv(self.X, self.y, cache_transformers=True, **options)
        parallel = fbcsp.nested_cv(self.X, self.y, n_jobs=2, **options)
        self.assertEqual(no_cache, cached)
        self.assertEqual(cached[:2], parallel[:2])
        self.assertEqual(cached[2]["folds"], parallel[2]["folds"])
        paired = fbcsp.nested_cv(self.X[:, :1], self.y, baseline=True, **options)
        baseline_grid = {k.replace("fbcsp__", "csp__"): v for k, v in self.grid.items()}
        reference = ws.nested_cv(self.X[:, 0], self.y, k_fold=2, inner_k_fold=2,
                                 param_grid=baseline_grid, return_details=True)
        self.assertEqual(paired[:2], reference[:2])
        self.assertEqual([f["best_params"] for f in paired[2]["folds"]],
                         [f["best_params"] for f in reference[2]["folds"]])

    def test_invalid_grid_or_folds_fail_before_fit(self):
        for grid in ({"fbcsp__reg": [-.1]}, {"fbcsp__log": [1]}, {"svm__C": [0]},
                     {"fbcsp__n_components": [5]}, {"svm__kernel": ["rbf"]}, {}):
            with self.subTest(grid=grid), patch.object(fbcsp.FBCSPTransformer, "fit") as fit:
                with self.assertRaises(ValueError):
                    fbcsp.nested_cv(self.X, self.y, param_grid=grid)
                fit.assert_not_called()
        with self.assertRaises(ValueError):
            fbcsp.nested_cv(self.X, self.y, k_fold=11)

    def test_actual_complete_current_12_candidate_grid(self):
        a, scores, details = fbcsp.nested_cv(self.X, self.y, k_fold=2, inner_k_fold=2,
                                            return_details=True)
        self.assertEqual(details["n_candidates"], 12)
        self.assertEqual(len(scores), 2)
        self.assertEqual(a, np.mean(scores))
        self.assertTrue(all(fold["best_params"] in list(ParameterGrid(fbcsp.FBCSP_PARAM_GRID))
                            for fold in details["folds"]))

    def test_default_ten_by_five_and_rank_deficient_input(self):
        low_rank = self.X.copy()
        low_rank[:, :, 3] = low_rank[:, :, 0] + low_rank[:, :, 1]
        grid = dict(self.grid, fbcsp__n_components=[4], svm__C=[1.])
        _, scores, details = fbcsp.nested_cv(low_rank, self.y, param_grid=grid, return_details=True)
        self.assertEqual(len(scores), 10)
        self.assertEqual(details["inner_k_fold"], 5)


class PublicationTest(Fixture):
    def test_round_trip_and_result_publication(self):
        report_path, paths = self.publication()
        before = {p: p.read_bytes() for p in paths}
        data, labels, metadata = fbcsp.load_fbcsp_data(paths[1])
        np.testing.assert_array_equal(data, self.X.astype(np.float32).astype(np.float64))
        self.assertFalse(data.flags.writeable)
        result = fbcsp_main.run_classification(report_path=report_path,
                                               output_dir=self.root / "result", k_fold=2,
                                               inner_k_fold=2, param_grid=self.grid)
        self.assertIn("baseline", result)
        self.assertEqual(json.loads((self.root / "result" / "classification_result.json").read_text()), result)
        for p, expected in before.items():
            self.assertEqual(p.read_bytes(), expected)
        with self.assertRaises(FileExistsError):
            fbcsp_main.run_classification(report_path=report_path, output_dir=self.root / "result")

    def test_contract_and_hash_reject_corruption(self):
        report_path, paths = self.publication()
        original = paths[1].read_bytes()
        for key, value in (("sampling_rate", -1.), ("axis_order", "wrong"), ("tmax", 8.),
                           ("source_raw_sha256", "wrong"), ("label_map", "{}")):
            paths[1].write_bytes(original)
            with h5py.File(paths[1], "r+") as stream:
                stream.attrs[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fbcsp.validate_fbcsp_hdf5(paths[1])
        paths[1].write_bytes(original)
        with h5py.File(paths[1], "r+") as stream:
            stream["clean_eeg"][0, 0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "SHA256"):
            fbcsp_main.load_publication(report_path)


if __name__ == "__main__":
    unittest.main()