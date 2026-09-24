"""Check the legacy OLS estimator, DNA normalization, and grouped aggregation."""

import unittest

import numpy as np
import pandas as pd
from scipy import sparse, stats

from compute_activities import feature_metadata
from compute_glm_activities import compute_glm_tables, fit_total_ols


class GLMActivityTests(unittest.TestCase):
    def test_matches_independent_linear_regression(self):
        x = np.arange(1., 9.)
        y = np.column_stack([2 + 3 * x, 4 + x + np.array([1, -1, 2, 0, -1, 0, 1, -2])])
        dna = np.array([100., 1000.])
        result = fit_total_ols(y, x, dna)
        for j in range(y.shape[1]):
            reference = stats.linregress(x, y[:, j])
            self.assertAlmostEqual(result.slope[j], reference.slope)
            self.assertAlmostEqual(result.intercept[j], reference.intercept)
            self.assertAlmostEqual(result.slope_se[j], reference.stderr)
            self.assertAlmostEqual(result.activity[j], reference.slope / np.log(dna[j]))
            self.assertAlmostEqual(result.activity_se[j], reference.stderr / np.log(dna[j]))
            self.assertAlmostEqual(result.log_activity[j], np.log(reference.slope / np.log(dna[j])))

    def test_intercept_is_fitted(self):
        x = np.arange(1., 10.)
        result = fit_total_ols((10 + 2 * x)[:, None], x, np.array([100.]))
        self.assertAlmostEqual(result.intercept[0], 10)
        self.assertAlmostEqual(result.slope[0], 2)

    def test_degenerate_inputs_are_flagged(self):
        y = np.column_stack([np.arange(5.), np.arange(5.)[::-1], np.ones(5)])
        x = np.arange(5.)
        result = fit_total_ols(y, x, np.array([1., 100., 100.]))
        self.assertEqual(result.fit_status.tolist(), ["dna_count_le1", "nonpositive_slope", "nonpositive_slope"])
        self.assertTrue(result.log_activity.isna().all())
        for values, predictor, status in ((y, np.ones(5), "constant_total_rna"),
                                          (y[:2], x[:2], "insufficient_cells")):
            result = fit_total_ols(values, predictor, np.full(3, 100.))
            self.assertTrue(result.fit_status.eq(status).all())
            self.assertTrue(result.log_activity.isna().all())

    def test_grouped_pooled_dna_alignment_and_sparse_equivalence(self):
        meta = feature_metadata(pd.Index(["CRE001_bar001", "CRE001_bar002", "barcode only_bar003"]))
        dna = pd.Series([100., 400., 20.], index=meta.index)
        counts = np.array([[2, 1, 0], [2, 3, 1], [4, 1, 4], [5, 4, 3],
                           [1, 0, 1], [3, 1, 0], [2, 3, 2], [4, 5, 2]], dtype=float)
        labels = pd.Series(pd.Categorical(["a"] * 4 + ["b"] * 4, categories=["a", "b", "unused"]))
        tables = compute_glm_tables(counts, labels, meta, dna.iloc[::-1])
        sparse_tables = compute_glm_tables(sparse.csr_matrix(counts), labels, meta, dna)
        for level in tables:
            pd.testing.assert_frame_equal(tables[level], sparse_tables[level])
        self.assertEqual(len(tables["barcode"]), 6)
        self.assertEqual(len(tables["cre"]), 4)
        row = tables["cre"].set_index(["group", "cre"]).loc[("a", "CRE001")]
        reference = stats.linregress(counts[:4].sum(1), counts[:4, :2].sum(1))
        self.assertAlmostEqual(row.activity, reference.slope / np.log(500))
        self.assertEqual(row.n_positive_cells, 4)
        self.assertEqual(row.dna_count, 500)
        summed_slopes = tables["barcode"].loc[lambda t: t.group.eq("a") & t.cre.eq("CRE001"), "slope"].sum()
        self.assertAlmostEqual(row.slope, summed_slopes)


if __name__ == "__main__":
    unittest.main()
