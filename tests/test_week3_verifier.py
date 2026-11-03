"""Actual selected K12 graph/lifecycle regressions; no numerical-proof claim.

Sample/enumeration comparisons below detect implementation mistakes.  Float64
allclose tolerances in those comparisons are not certificate acceptance rules.
"""

import copy
from dataclasses import FrozenInstanceError, replace
import itertools
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from ecdr.week3_verifier import (
    ActualGraphIBP, CandidateBox, CHECKPOINT_SHA256, FrozenK12Reference,
    IdentityError, PREPROCESSING_ID, RuntimeGate, UnsupportedGraphError,
    check_certificate, diagonal_box, fixed_overlap_wave, interval_layer,
)


class OperatorTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def assert_samples_contained(self, layer, lower, upper, n=12):
        lo, hi = interval_layer(layer, lower, upper)
        for _ in range(n):
            x = lower + torch.rand_like(lower) * (upper - lower)
            y = layer(x)
            self.assertTrue(bool(torch.all(y >= lo - 1e-12)))
            self.assertTrue(bool(torch.all(y <= hi + 1e-12)))

    def test_padded_signed_conv(self):
        conv = nn.Conv1d(2, 3, 3, padding=1).double().eval()
        lower = -torch.rand(1, 2, 5, dtype=torch.float64)
        self.assert_samples_contained(conv, lower, lower + 1)

    def test_grouped_dilated_conv(self):
        conv = nn.Conv1d(2, 2, 3, dilation=2, padding=2, groups=2).double().eval()
        lower = -torch.rand(1, 2, 5, dtype=torch.float64)
        self.assert_samples_contained(conv, lower, lower + 1)

    def test_batchnorm_negative_gamma(self):
        bn = nn.BatchNorm1d(2).double().eval()
        with torch.no_grad():
            bn.weight[:] = torch.tensor([-2.0, 3.0])
            bn.bias[:] = torch.tensor([0.25, -0.5])
            bn.running_mean[:] = torch.tensor([1.0, -2.0])
            bn.running_var[:] = torch.tensor([4.0, 0.25])
        lower = torch.tensor([[[-2.0, -1.0], [1.0, 2.0]]], dtype=torch.float64)
        self.assert_samples_contained(bn, lower, lower + 1)

    def test_batchnorm_without_affine(self):
        bn = nn.BatchNorm1d(2, affine=False).double().eval()
        lower = -torch.rand(1, 2, 5, dtype=torch.float64)
        self.assert_samples_contained(bn, lower, lower + 1)

    def test_reject_training_or_untracked_batchnorm(self):
        lower = torch.zeros(1, 2, 4, dtype=torch.float64)
        for bn in (nn.BatchNorm1d(2).double(), nn.BatchNorm1d(2, track_running_stats=False).double().eval()):
            with self.assertRaises(UnsupportedGraphError):
                interval_layer(bn, lower, lower)

    def test_maxpool_negative_inputs_padding_and_ceil(self):
        # PyTorch max-pool padding is negative infinity, not numeric zero.
        pool = nn.MaxPool1d(3, stride=2, padding=1, ceil_mode=True)
        lower = torch.tensor([[[-5., -4., -3., -2., -1.]]], dtype=torch.float64)
        lo, hi = interval_layer(pool, lower, lower + 0.25)
        self.assertTrue(bool(torch.all(hi < 0)))
        self.assert_samples_contained(pool, lower, lower + 0.25)

    def test_flatten_relu_and_linear(self):
        lower = -torch.rand(1, 2, 4, dtype=torch.float64)
        self.assert_samples_contained(nn.Flatten(), lower, lower + 1)
        self.assert_samples_contained(nn.ReLU(), lower, lower + 1)
        lower = lower.flatten(1)
        self.assert_samples_contained(nn.Linear(8, 3).double().eval(), lower, lower + 1)

    def test_reject_unsupported_operator_and_nonfinite_intervals(self):
        x = torch.zeros(1, 2, 4, dtype=torch.float64)
        with self.assertRaises(UnsupportedGraphError):
            interval_layer(nn.Sigmoid(), x, x)
        with self.assertRaises(ValueError):
            interval_layer(nn.ReLU(), x, x + float("inf"))
        with self.assertRaises(ValueError):
            interval_layer(nn.ReLU(), x + 1, x)
        with self.assertRaises(ValueError):
            interval_layer(nn.ReLU(), x.float(), x.float())

    def test_direct_margin_dependency_cancellation(self):
        # Equal logit weights cancel identically in a direct specification.
        lower = torch.tensor([[-2., -3.]], dtype=torch.float64)
        upper = torch.tensor([[4., 5.]], dtype=torch.float64)
        head = nn.Linear(2, 2).double().requires_grad_(False)
        with torch.no_grad():
            head.weight[:] = torch.tensor([[2., -1.], [2., -1.]])
            head.bias[:] = torch.tensor([1., 0.])
        lo, hi = interval_layer(head, lower, upper)
        separate = float(lo[0, 0] - hi[0, 1])
        direct = (head.weight[0] - head.weight[1]) @ lower[0] + head.bias[0] - head.bias[1]
        self.assertEqual(float(direct), 1.0)
        self.assertLess(separate, 0)
        # Enumerate this tiny affine box's vertices independently.
        for vertex in itertools.product(*zip(lower[0].tolist(), upper[0].tolist())):
            logits = head(torch.tensor(vertex, dtype=torch.float64))
            self.assertEqual(float(logits[0] - logits[1]), 1.0)


class DomainTests(unittest.TestCase):
    def test_zero_budget_boundary_and_no_epsilon(self):
        center = np.zeros((6, 128), np.float32)
        box = diagonal_box(center, 0)
        self.assertEqual(box.free_dimensions, 0)
        self.assertTrue(box.contains(center))
        outside = center.copy()
        outside[0, 0] = np.nextafter(np.float32(0), np.float32(1))
        self.assertFalse(box.contains(outside))
        box = diagonal_box(center, 0.25)
        self.assertTrue(box.contains(np.full_like(center, 0.25)))
        self.assertTrue(box.contains(np.full_like(center, -0.25)))
        self.assertFalse(box.contains(np.full_like(center, np.nextafter(np.float32(.25), np.float32(1)))))

    def test_budget_broadcast_and_invalid_domains(self):
        center = np.zeros((6, 128), np.float32)
        self.assertEqual(diagonal_box(center, np.arange(6)).free_dimensions, 5 * 128)
        for radius in (-1., float("nan"), float("inf"), np.zeros((5, 128))):
            with self.assertRaises(ValueError):
                diagonal_box(center, radius)
        with self.assertRaises(ValueError):
            CandidateBox((0.,) * 768, (-1.,) * 768, "diagonal_box", 768)
        with self.assertRaises(ValueError):
            CandidateBox((0.,) * 768, (0.,) * 768, "diagonal_box", 1)

    def test_fixed_overlap_wave_has_384_variables(self):
        center = np.arange(768, dtype=np.float32).reshape(6, 128)
        wave = fixed_overlap_wave(center, .25)
        lo, hi = wave.arrays()
        self.assertEqual(wave.free_dimensions, 384)
        np.testing.assert_array_equal(lo[:, :64], center[:, 64:])
        np.testing.assert_array_equal(hi[:, :64], center[:, 64:])
        query = np.concatenate((center[:, 64:], center[:, 64:]), axis=1)
        self.assertTrue(wave.contains(query))
        query[0, 0] += 1
        self.assertFalse(wave.contains(query))

    def test_domain_storage_is_immutable_and_unaliased(self):
        values = [0.] * 768
        box = CandidateBox(values, values, "diagonal_box", 0)
        initial_id = box.domain_id
        values[0] = 10.
        self.assertEqual(box.lower[0], 0.)
        self.assertEqual(box.domain_id, initial_id)
        lo, hi = box.arrays()
        lo[0, 0] = 3
        self.assertEqual(box.lower[0], 0.)
        with self.assertRaises(FrozenInstanceError):
            box.kind = "wrong"


class FrozenActualGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference = FrozenK12Reference.load()
        cls.verifier = ActualGraphIBP(cls.reference)
        cls.raw = np.zeros((6, 128), np.float32)
        cls.build = cls.verifier.build(cls.raw, 0., recording_id="r1", step_idx=0, max_age=3)
        if not cls.build.conditional_positive:
            raise AssertionError("frozen zero-radius regression fixture unexpectedly inconclusive")

    def test_exact_selected_k12_identity_and_normalization(self):
        ref = self.reference
        self.assertEqual(ref.identity.checkpoint_sha256, CHECKPOINT_SHA256)
        self.assertEqual(ref.identity.preprocessing_id, PREPROCESSING_ID)
        self.assertEqual(len(ref.class_order), 12)
        self.assertEqual(sum(p.numel() for p in ref.model.parameters()), 283692)
        raw = np.arange(768, dtype=np.float32).reshape(6, 128) / 100
        expected = ((raw - ref.mean[:, None]) / ref.std[:, None]).astype(np.float32)
        np.testing.assert_array_equal(ref.normalize(raw), expected)

    def test_zero_radius_eleven_margins_match_double_point_diagnostic(self):
        build = self.build
        self.assertEqual(len(build.competitors), 11)
        self.assertEqual(build.competitors, tuple(j for j in range(12) if j != build.target_class))
        with torch.inference_mode():
            logits = self.verifier._model(torch.from_numpy(self.reference.normalize(self.raw)[None]).double()).numpy()[0]
        expected = logits[build.target_class] - logits[list(build.competitors)]
        np.testing.assert_allclose(build.lower_margins, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(build.numerical_status, "NOT ESTABLISHED")
        self.assertFalse(build.formal_reuse_enabled)
        self.assertFalse(build.certificate.formal_reuse_enabled)

    def test_actual_graph_small_box_sample_consistency_and_direct_tightness(self):
        domain = diagonal_box(self.reference.normalize(self.raw), 1e-5)
        target = self.build.target_class
        direct, separate = self.verifier.bounds(domain, target)
        self.assertTrue(np.all(direct >= separate - 1e-10))
        rng = np.random.default_rng(8)
        lo, hi = domain.arrays()
        for _ in range(8):
            z = rng.uniform(lo, hi)
            with torch.inference_mode():
                logits = self.verifier._model(torch.from_numpy(z[None])).numpy()[0]
            actual = logits[target] - logits[list(self.build.competitors)]
            self.assertTrue(np.all(actual >= direct - 1e-10))

    def test_invalid_raw_rejected_without_inference_or_state_mutation(self):
        gate = RuntimeGate(self.reference, self.verifier)
        for raw in (np.zeros((1, 6, 128)), np.full((6, 128), np.nan),
                    np.full((6, 128), np.inf), np.full((6, 128), 1e300),
                    np.full((6, 128), 1+1j)):
            with self.assertRaises(ValueError):
                gate.predict(raw, recording_id="r1", step_idx=0)
            self.assertIsNone(gate.last_step)

    def test_normalization_overflow_does_not_reset_existing_state(self):
        gate = RuntimeGate(self.reference, self.verifier)
        gate.predict(self.raw, recording_id="r1", step_idx=0, radius=0.)
        cert_id = gate.certificate.certificate_id
        with self.assertRaises(ValueError):
            gate.predict(np.full((6, 128), np.finfo(np.float32).max), recording_id="r2", step_idx=0)
        self.assertEqual(gate.recording_id, "r1")
        self.assertEqual(gate.last_step, 0)
        self.assertEqual(gate.certificate.certificate_id, cert_id)

    def test_boolean_or_negative_class_and_location_rejected(self):
        domain = diagonal_box(self.reference.normalize(self.raw), 0.)
        for target in (True, np.bool_(False), -1, 12, 0.5):
            with self.assertRaises(ValueError):
                self.verifier.bounds(domain, target)
        for step in (True, np.bool_(False), -1, 0.5):
            with self.assertRaises(ValueError):
                self.verifier.build(self.raw, 0., recording_id="r1", step_idx=step)

    def test_tied_and_tiny_anchor_margin_unit_branches(self):
        # Deliberate mocks exercise rejection branches; these are not claimed as
        # observed logits or reference execution evidence for the frozen model.
        tied = np.zeros(12, dtype=np.float32)
        with mock.patch.object(self.reference, "infer", return_value=(tied, 0.)):
            result = self.verifier.build(self.raw, 0., recording_id="r1", step_idx=0)
        self.assertEqual(result.reason, "anchor_tie")
        self.assertIsNone(result.certificate)
        tiny = tied.copy()
        tiny[0] = np.nextafter(np.float32(0), np.float32(1))
        with mock.patch.object(self.reference, "infer", return_value=(tiny, 0.)):
            with mock.patch.object(self.verifier, "bounds", return_value=(np.full(11, 1e-300), np.full(11, 1e-300))):
                result = self.verifier.build(self.raw, 0., recording_id="r1", step_idx=np.int64(0))
        self.assertTrue(result.conditional_positive)
        self.assertFalse(result.formal_reuse_enabled)
        self.assertIs(type(result.certificate.anchor_step), int)

    def test_mutated_weights_buffers_class_order_and_preprocessing_rejected(self):
        for kind in ("weights", "buffer", "classes", "normalization", "training", "dtype", "operator"):
            ref = copy.deepcopy(self.reference)
            with torch.no_grad():
                if kind == "weights":
                    ref.model.head[-1].bias[0] += 1
                elif kind == "buffer":
                    ref.model.conv[1].running_mean[0] += 1
                elif kind == "classes":
                    ref.class_order = tuple(reversed(ref.class_order))
                elif kind == "normalization":
                    ref.mean[0] += .01
                elif kind == "training":
                    ref.model.train()
                elif kind == "dtype":
                    ref.model.double()
                elif kind == "operator":
                    ref.model.conv[2] = nn.Sigmoid()
            with self.subTest(kind=kind), self.assertRaises(IdentityError):
                ref.infer(self.raw)

    def test_lifecycle_negative_identity_anchor_expiry_and_specifications(self):
        cert = self.build.certificate
        cases = (
            (replace(cert, preprocessing_id="wrong"), "r1", 1, "identity_mismatch"),
            (replace(cert, checkpoint_sha256="wrong"), "r1", 1, "identity_mismatch"),
            (replace(cert, class_order=tuple(reversed(cert.class_order))), "r1", 1, "identity_mismatch"),
            (replace(cert, graph_id="wrong"), "r1", 1, "identity_mismatch"),
            (cert, "r2", 1, "recording_reset"),
            (cert, "r1", 4, "stale_or_expired_anchor"),
            (cert, "r1", 0, "stale_or_expired_anchor"),
            (replace(cert, lower_margins=(0.,) * 11), "r1", 1, "invalid_margin_specification"),
            (replace(cert, lower_margins=(float("nan"),) * 11), "r1", 1, "invalid_margin_specification"),
            (replace(cert, competitors=(0,)), "r1", 1, "invalid_margin_specification"),
            (replace(cert, lifecycle="ACTIVE"), "r1", 1, "invalid_certificate_state_or_method"),
        )
        for candidate, recording, step, reason in cases:
            result = check_certificate(self.reference, candidate, self.raw, recording_id=recording, step_idx=step)
            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, reason)
        wrong_anchor = self.raw.copy()
        wrong_anchor[0, 0] = 1
        result = check_certificate(self.reference, cert, self.raw, recording_id="r1", step_idx=1, expected_anchor_raw=wrong_anchor)
        self.assertEqual(result.reason, "anchor_mismatch")

    def test_numerical_gate_cannot_be_bypassed_by_positive_or_forged_certificate(self):
        cert = self.build.certificate
        for cert in (cert, replace(cert, lower_margins=(1e300,) * 11, numerical_status="ESTABLISHED")):
            gate = check_certificate(self.reference, cert, self.raw, recording_id="r1", step_idx=1)
            self.assertTrue(gate.is_member)
            self.assertFalse(gate.accepted)
            self.assertEqual(gate.reason, "numerical_guarantee_unavailable")

    def test_runtime_fixed_anchor_resets_fallback_and_expiry(self):
        runtime = RuntimeGate(self.reference, self.verifier)
        first = runtime.predict(self.raw, recording_id="r1", step_idx=0, radius=0., max_age=3)
        original_id = runtime.certificate.certificate_id
        for step in (1, 2):
            result = runtime.predict(self.raw, recording_id="r1", step_idx=step, radius=0., max_age=3)
            self.assertFalse(result.is_reuse)
            self.assertTrue(result.is_member)
            self.assertEqual(runtime.certificate.certificate_id, original_id)
            self.assertEqual(runtime.certificate.anchor_step, 0)
            self.assertEqual(result.pred_class, first.pred_class)
        result = runtime.predict(self.raw, recording_id="r1", step_idx=4, radius=0.)
        self.assertEqual(result.reason, "discontinuity_reset")
        self.assertEqual(runtime.certificate.anchor_step, 4)
        result = runtime.predict(self.raw, recording_id="r2", step_idx=0)
        self.assertEqual(result.reason, "recording_reset")
        self.assertIsNone(runtime.certificate)
        runtime.reset()
        self.assertIsNone(runtime.last_step)

    def test_runtime_inconclusive_build_cannot_create_certificate(self):
        runtime = RuntimeGate(self.reference, self.verifier)
        result = runtime.predict(self.raw, recording_id="r1", step_idx=0, radius=1.)
        self.assertFalse(result.is_reuse)
        self.assertFalse(result.build_result.conditional_positive)
        self.assertIsNone(runtime.certificate)
        reference_logits, _ = self.reference.infer(self.raw)
        self.assertEqual(result.pred_class, int(reference_logits.argmax()))

    def test_bound_snapshot_mutation_fails_closed(self):
        verifier = ActualGraphIBP(self.reference)
        with torch.no_grad():
            verifier._model.head[-1].bias[0] += 1
        result = verifier.build(self.raw, 0., recording_id="r1", step_idx=0)
        self.assertEqual(result.status, "FAILED")
        self.assertIsNone(result.certificate)
        self.assertFalse(result.formal_reuse_enabled)

    def test_forward_hooks_and_child_forward_override_invalidate_identity(self):
        for kind in ("hook", "prehook", "child_forward"):
            ref = copy.deepcopy(self.reference)
            if kind == "hook":
                ref.model.head[-1].register_forward_hook(lambda module, args, output: output + 1)
            elif kind == "prehook":
                ref.model.conv[0].register_forward_pre_hook(lambda module, args: (args[0] + 1,))
            else:
                ref.model.head[-1].forward = lambda x: torch.zeros((len(x), 12))
            with self.subTest(kind=kind), self.assertRaises(IdentityError):
                ref.infer(self.raw)


if __name__ == "__main__":
    unittest.main()
