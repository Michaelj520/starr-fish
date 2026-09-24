"""Small hand-calculated examples for the activity definition and aggregation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy import sparse

from compute_activities import compute_tables, feature_metadata, load_dna_counts, log_fold_change


class ActivityTests(unittest.TestCase):
    def setUp(self):
        # Unequal barcode multiplicities ensure CRE sums cannot masquerade as means.
        self.metadata = feature_metadata(pd.Index([
            "CRE001_bar001", "CRE001_bar002", "CRE002_bar003",
            "barcode only_bar004", "barcode only_bar005",
        ]))
        self.counts = np.array([[8, 0, 4, 1, 3], [0, 4, 2, 3, 1], [0, 0, 1, 0, 0]], dtype=float)
        self.labels = pd.Series(pd.Categorical(["a", "a", "b"], categories=["a", "b", "unused"]))
        self.dna = pd.Series([4., 4., 6., 2., 10.], index=self.metadata.index)

    def test_hand_computed_rna_dna_ratios(self):
        tables = compute_tables(self.counts, self.labels, self.metadata, self.dna)
        barcode = tables["barcode"].set_index(["group", "feature"])
        cre = tables["cre"].set_index(["group", "feature"])
        # Group a RNA totals [8,4,6,4,4], DNA [4,4,6,2,10]; both libraries total 26.
        self.assertAlmostEqual(barcode.loc[("a", "CRE001_bar001"), "log_fold_change"], np.log(2))
        self.assertAlmostEqual(cre.loc[("a", "CRE001"), "log_fold_change"], np.log(1.5))
        self.assertEqual(cre.loc[("a", "CRE001"), "n_positive_cells"], 2)
        self.assertEqual(cre.loc[("a", "CRE001"), "total_counts"], 12)
        self.assertAlmostEqual(cre.loc[("a", "barcode only"), "log_fold_change"], np.log(8 / 12))
        self.assertEqual(cre.loc[("a", "CRE002"), "log_fold_change"], 0)
        self.assertAlmostEqual(cre.loc[("a", "CRE001"), "mean_barcode_log_fold_change"], np.log(2) / 2)
        self.assertAlmostEqual(cre.loc[("b", "CRE002"), "log_fold_change"], np.log(26 / 6))
        self.assertTrue(np.isneginf(cre.loc[("b", "CRE001"), "log_fold_change"]))
        self.assertNotIn("unused", cre.index.get_level_values("group"))

    def test_depth_scaling_and_dna_order_do_not_change_activity(self):
        baseline = compute_tables(self.counts, self.labels, self.metadata, self.dna)
        scaled = compute_tables(7 * self.counts, self.labels, self.metadata, (3 * self.dna).iloc[::-1])
        for key in baseline:
            np.testing.assert_allclose(baseline[key].log_fold_change, scaled[key].log_fold_change, atol=1e-12)

    def test_dna_changes_barcode_activity(self):
        dna = self.dna.copy()
        dna.iloc[0] *= 2
        table = compute_tables(self.counts, self.labels, self.metadata, dna)["barcode"]
        row = table.loc[table.group.eq("a") & table.feature.eq("CRE001_bar001")].iloc[0]
        self.assertAlmostEqual(row.log_fold_change, np.log((8 / 26) / (8 / 30)))

    def test_sparse_and_dense_agree(self):
        dense = compute_tables(self.counts, self.labels, self.metadata, self.dna)
        csr = compute_tables(sparse.csr_matrix(self.counts), self.labels, self.metadata, self.dna)
        for key in dense:
            pd.testing.assert_frame_equal(dense[key], csr[key])

    def test_zero_policy_and_base(self):
        result = log_fold_change(np.array([[0., 4., 1.]]), np.array([2., 2., 0.]), 0, 2)
        self.assertTrue(np.isneginf(result[0, 0]))
        self.assertEqual(result[0, 1], 1)
        self.assertTrue(np.isnan(result[0, 2]))
        result = log_fold_change(np.zeros((1, 2)), np.array([2., 0.]), 1, 2)
        self.assertAlmostEqual(result[0, 0], np.log2(1 / 3))
        self.assertTrue(np.isnan(result[0, 1]))

    def test_zero_dna_and_empty_rna_group(self):
        dna = self.dna.copy()
        dna.iloc[0] = 0
        counts = self.counts.copy()
        counts[2] = 0
        tables = compute_tables(counts, self.labels, self.metadata, dna, pseudocount=1)
        barcode = tables["barcode"]
        self.assertTrue(barcode.loc[barcode.feature.eq("CRE001_bar001"), "log_fold_change"].isna().all())
        self.assertTrue(barcode.loc[barcode.group.eq("b"), "log_fold_change"].isna().all())
        cre = tables["cre"]
        self.assertTrue(cre.loc[cre.cre.eq("CRE001"), "mean_barcode_log_fold_change"].isna().all())

    def test_dna_file_alignment_alias_and_rejections(self):
        dna = pd.DataFrame({"element": self.metadata.index.str.replace("barcode only_", "barcode_only_"),
                            "DNA_count": self.dna.to_numpy()}).iloc[::-1]
        with TemporaryDirectory() as temp:
            path = Path(temp) / "dna.csv"
            dna.to_csv(path, index=False)
            loaded = load_dna_counts(path, self.metadata.index)
            np.testing.assert_array_equal(loaded.DNA_count, self.dna)
            self.assertAlmostEqual(loaded.DNA_CPM.sum(), 1e6)
            for bad in (dna.iloc[:-1], pd.concat([dna, dna.iloc[:1]]),
                        dna.assign(DNA_count=-1), dna.assign(DNA_count=np.nan),
                        dna.assign(DNA_count=0), dna.assign(DNA_count=1.5)):
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_dna_counts(path, self.metadata.index)

    def test_malformed_or_duplicate_barcodes_fail(self):
        for names in (["CRE001"], ["CRE001_bar001", "CRE002_bar001"]):
            with self.assertRaises(ValueError):
                feature_metadata(pd.Index(names))


if __name__ == "__main__":
    unittest.main()
