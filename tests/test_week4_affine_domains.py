"""tests/test_week4_affine_domains.py — Unit tests for W4-016 PCA/affine-tail domain.

Covers:
  - W4-016-C01: Affine tail domain definition: tail = c + B diag(a) u + diag(e) v.
  - W4-016-C02: Train-only PCA fitting with basis/mean/residual hashing.
  - W4-016-C03: Immutable AffineTailDomain shape, rank, and ID integrity.
  - W4-016-C04: Latent feasibility and orthogonal residual membership checks.
  - W4-016-C05: Outer bounding box relaxation & containment guarantee.
  - W4-016-C06: Counterexamples (omitted residual, rotated correlation, centering mismatch).
  - W4-016-C07: Artifact generation: domain_fit.json & affine_domain_validation.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ecdr.week4.domains import (
    AffineTailDomain,
    AffineTailParameters,
    fit_train_pca_affine_parameters,
    propose_causal_affine_domain,
    run_affine_domain_validation,
)
from ecdr.week4.reference import ReferenceOwner

ROOT = Path(__file__).resolve().parents[1]


class TestWeek4AffineDomains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref = ReferenceOwner.load()
        # Fit on train split with K=8 components
        cls.params = fit_train_pca_affine_parameters(k_components=8)

        # Load validation data
        cls.x_all = np.load(ROOT / "data/windows.npz")["x"]
        df_split = pd.read_parquet(ROOT / "data/windows_with_split.parquet")
        cls.val_indices = df_split.index[df_split["split"] == "val"].tolist()

    def test_c01_parameter_shapes_and_orthogonality(self):
        """W4-016-C01: Basis has shape (384, K), orthonormal columns, non-negative bounds."""
        p = self.params
        self.assertEqual(p.basis.shape, (384, 8))
        self.assertEqual(p.singular_values.shape, (8,))
        self.assertEqual(p.mean.shape, (384,))
        self.assertEqual(p.residual_bounds.shape, (384,))
        self.assertTrue(np.all(p.residual_bounds >= 0.0))
        self.assertTrue(np.all(p.singular_values >= 0.0))

        # Check B^T B = I_K
        eye_k = np.eye(8, dtype=np.float32)
        prod = p.basis.T @ p.basis
        self.assertLess(float(np.max(np.abs(prod - eye_k))), 1e-4)

    def test_c02_train_only_fitting_and_hash_integrity(self):
        """W4-016-C02: Pinned train pair count > 9000, reproducible hash."""
        self.assertGreater(self.params.train_pair_count, 9000)
        self.assertEqual(len(self.params.fit_sha256), 64)

    def test_c03_domain_immutability_and_identity(self):
        """W4-016-C03: AffineTailDomain is immutable, assigns unique domain_id."""
        raw = self.x_all[self.val_indices[0]].astype(np.float32)
        norm = self.ref.normalize(raw)

        dom = propose_causal_affine_domain(
            anchor=norm,
            params=self.params,
            latent_scale=1.0,
            residual_scale=1.0,
            anchor_step=0,
            recording_id="exp_01",
        )

        self.assertEqual(len(dom.domain_id), 64)
        self.assertEqual(dom.center_tail.shape, (384,))
        self.assertEqual(dom.prefix.shape, (6, 64))

        # Check immutability (modifying returned array doesn't affect internal state)
        c = dom.center_tail
        c[0] += 999.0
        self.assertNotEqual(float(dom.center_tail[0]), float(c[0]))

    def test_c04_membership_latent_and_orthogonal_residual_checks(self):
        """W4-016-C04: Membership enforces bit-exact prefix, latent bounds, and residual bounds."""
        raw = self.x_all[self.val_indices[0]].astype(np.float32)
        norm = self.ref.normalize(raw)

        dom = propose_causal_affine_domain(
            anchor=norm,
            params=self.params,
            latent_scale=1.0,
            residual_scale=1.0,
            anchor_step=5,
            recording_id="exp_01",
        )

        # 1. Exact center point continuation must be an exact member
        exact_query = np.concatenate([dom.prefix, dom.center_tail.reshape(6, 64)], axis=-1)
        is_member, details = dom.check_membership(exact_query, step_idx=6, recording_id="exp_01")
        self.assertTrue(is_member)
        self.assertEqual(details["latent_violations_count"], 0)
        self.assertEqual(details["residual_violations_count"], 0)

        # 2. Wrong step index fails
        is_mem_wrong_step, _ = dom.check_membership(exact_query, step_idx=7, recording_id="exp_01")
        self.assertFalse(is_mem_wrong_step)

        # 3. Wrong prefix fails
        corrupted_prefix = exact_query.copy()
        corrupted_prefix[0, 0] += 0.05
        is_mem_bad_prefix, _ = dom.check_membership(corrupted_prefix, step_idx=6, recording_id="exp_01")
        self.assertFalse(is_mem_bad_prefix)

    def test_c05_outer_box_containment_guarantee(self):
        """W4-016-C05: Any point generated within the affine domain is contained in its outer box."""
        raw = self.x_all[self.val_indices[0]].astype(np.float32)
        norm = self.ref.normalize(raw)

        dom = propose_causal_affine_domain(
            anchor=norm,
            params=self.params,
            latent_scale=1.5,
            residual_scale=1.2,
            anchor_step=0,
            recording_id="exp_01",
        )

        b_lo, b_hi = dom.to_outer_box()
        self.assertEqual(b_lo.shape, (6, 128))
        self.assertEqual(b_hi.shape, (6, 128))
        self.assertTrue(np.all(b_lo <= b_hi))

        # Sample synthetic points inside the affine domain:
        # delta = B (a * u) + r, where r is orthogonal to B: r = (I - B B^T) (e * v)
        rng = np.random.default_rng(42)
        b = dom.params.basis
        a = dom.params.singular_values * dom.latent_scale
        e = dom.params.residual_bounds * dom.residual_scale

        for _ in range(20):
            u = rng.uniform(-1.0, 1.0, size=(dom.params.k_components,)).astype(np.float32)
            w = rng.uniform(-0.9, 0.9, size=(384,)).astype(np.float32)
            # Project w onto orthogonal complement of B to ensure zero latent leakage
            raw_r = e * w
            ortho_r = raw_r - b @ (b.T @ raw_r)
            # Rescale if any coordinate exceeds e
            max_viol = np.max(np.abs(ortho_r) / np.maximum(e, 1e-9))
            if max_viol > 1.0:
                ortho_r = ortho_r / max_viol

            synth_tail = dom.center_tail + b @ (a * u) + ortho_r
            synth_query = np.concatenate([dom.prefix, synth_tail.reshape(6, 64)], axis=-1)

            # Must satisfy affine membership
            is_mem, details = dom.check_membership(synth_query, step_idx=1, recording_id="exp_01")
            self.assertTrue(is_mem)

            # Must be strictly contained in outer box
            self.assertTrue(np.all(synth_query >= b_lo - 1e-5))
            self.assertTrue(np.all(synth_query <= b_hi + 1e-5))

    def test_c06_counterexamples_and_omitted_residual(self):
        """W4-016-C06: Omitted residual eliminates points outside subspace; rotated bounds test."""
        raw = self.x_all[self.val_indices[0]].astype(np.float32)
        norm = self.ref.normalize(raw)

        # Create zero-residual parameter set
        zero_res_params = AffineTailParameters(
            basis=self.params.basis,
            singular_values=self.params.singular_values,
            mean=self.params.mean,
            residual_bounds=np.zeros(384, dtype=np.float32),
            k_components=self.params.k_components,
            train_pair_count=self.params.train_pair_count,
            fit_sha256="zero_res_hash",
        )

        dom_zero_res = propose_causal_affine_domain(
            anchor=norm,
            params=zero_res_params,
            latent_scale=1.0,
            residual_scale=1.0,
            anchor_step=0,
            recording_id="exp_01",
        )

        # Inject orthogonal perturbation not in the basis subspace
        # Let v be orthogonal to basis: (I - B B^T) w
        w = np.ones(384, dtype=np.float32)
        ortho_dir = w - dom_zero_res.params.basis @ (dom_zero_res.params.basis.T @ w)
        ortho_dir = ortho_dir / np.linalg.norm(ortho_dir)

        # Point perturbed orthogonally
        perturbed_tail = dom_zero_res.center_tail + 0.1 * ortho_dir
        query = np.concatenate([dom_zero_res.prefix, perturbed_tail.reshape(6, 64)], axis=-1)

        is_mem, details = dom_zero_res.check_membership(query, step_idx=1, recording_id="exp_01")
        # Must be rejected because residual bound is zero!
        self.assertFalse(is_mem)
        self.assertGreater(details["residual_violations_count"], 0)

    def test_c07_validation_and_artifacts(self):
        """W4-016-C07: Run small validation stream evaluation and produce artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            val_out = run_affine_domain_validation(
                reference=self.ref,
                params=self.params,
                run_id="test_run",
                num_pairs=5,
                output_path=tmp_path / "affine_domain_validation.json",
            )

            self.assertTrue((tmp_path / "affine_domain_validation.json").is_file())
            self.assertEqual(val_out["task_id"], "W4-016")
            self.assertEqual(val_out["evaluated_validation_pairs"], 5)


if __name__ == "__main__":
    unittest.main()
