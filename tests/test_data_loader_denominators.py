"""test_data_loader_denominators.py — Test data loader contracts, split counts, and corrupted guards.

Authoritative source: G2_CLASS_DECISION.md and independent audit requirements.
"""

import unittest
import numpy as np

from ecdr.data import (
    FULL_STREAM_DENOMINATORS,
    load_full_manifest_with_splits,
    load_full_stream,
    load_raw_windows,
    load_supervised_dataset,
)


class TestDataLoaderDenominators(unittest.TestCase):
    def test_full_stream_denominators(self):
        # 1. Manifest level
        df = load_full_manifest_with_splits()
        self.assertEqual(len(df), FULL_STREAM_DENOMINATORS["total"])
        self.assertEqual(int((df["split"] == "train").sum()), 9978)
        self.assertEqual(int((df["split"] == "val").sum()), 2422)
        self.assertEqual(int((df["split"] == "test").sum()), 5053)

        # 2. Global row_id continuity
        np.testing.assert_array_equal(df["global_row_id"].to_numpy(), np.arange(17453))

        # 3. Stream loader level (no label filtering)
        x_te, df_te = load_full_stream("test")
        self.assertEqual(len(x_te), 5053)
        self.assertEqual(len(df_te), 5053)

        x_va, df_va = load_full_stream("val")
        self.assertEqual(len(x_va), 2422)
        self.assertEqual(len(df_va), 2422)

        x_tr, df_tr = load_full_stream("train")
        self.assertEqual(len(x_tr), 9978)
        self.assertEqual(len(df_tr), 9978)

    def test_supervised_dataset_counts(self):
        # Majority policy: pure+majority excluding ambiguous
        x_tr_maj, y_tr_maj, df_tr_maj = load_supervised_dataset("train", label_policy="majority", num_classes=12)
        self.assertEqual(len(x_tr_maj), 6955)

        x_va_maj, y_va_maj, df_va_maj = load_supervised_dataset("val", label_policy="majority", num_classes=12)
        self.assertEqual(len(x_va_maj), 1751)

        x_te_maj, y_te_maj, df_te_maj = load_supervised_dataset("test", label_policy="majority", num_classes=12)
        self.assertEqual(len(x_te_maj), 3572)

        # Ambiguous windows must be strictly zero in filtered dataframe
        self.assertFalse((df_tr_maj["label_policy"] == "ambiguous").any())
        self.assertFalse((df_va_maj["label_policy"] == "ambiguous").any())
        self.assertFalse((df_te_maj["label_policy"] == "ambiguous").any())

    def test_pairwise_disjoint_subjects(self):
        df = load_full_manifest_with_splits()
        tr_subjs = set(df[df["split"] == "train"]["subject"].unique())
        va_subjs = set(df[df["split"] == "val"]["subject"].unique())
        te_subjs = set(df[df["split"] == "test"]["subject"].unique())

        self.assertEqual(len(tr_subjs & va_subjs), 0)
        self.assertEqual(len(tr_subjs & te_subjs), 0)
        self.assertEqual(len(va_subjs & te_subjs), 0)

    def test_recording_single_split_and_subject(self):
        df = load_full_manifest_with_splits()
        self.assertTrue((df.groupby("exp")["split"].nunique() == 1).all())
        self.assertTrue((df.groupby("exp")["subject"].nunique() == 1).all())

    def test_raw_windows_finiteness(self):
        x = load_raw_windows()
        self.assertEqual(x.shape, (17453, 6, 128))
        self.assertTrue(np.all(np.isfinite(x)))


if __name__ == "__main__":
    unittest.main()
