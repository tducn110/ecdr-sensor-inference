"""replay.py — Chronological Streaming Replay Engine with Trusted Anchor Lifecycle.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Section 11)
and 07_G3_DOMAIN_CONTRACT_VI.md (Section 6).

Protocol requirements:
1. Strict recording reset at recording boundary (exp_id change) or sequence gap.
2. Trusted-anchor lifecycle: certificates are bound to exact anchor window and recording.
3. Candidate domain membership checked conservatively against exact stored interval endpoints.
4. Accepted queries reuse certified target class without re-evaluating backbone Conv layers.
5. Shadow reference agreement: reference forward runs in background to verify agreement.
6. Full cost instrumentation: wall-clock measured around complete operational path,
   excluding only separately measured shadow reference execution.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ecdr.bounds import (
    compute_diagonal_box_certificate,
    compute_wave_domain_certificate,
)
from ecdr.certificate import Certificate, CertificateManager, compute_window_hash
from ecdr.incremental import IncrementalEngine
from ecdr.instrumentation import CostInstrumentor, QueryRecord
from ecdr.intervals import Interval
from ecdr.membership import (
    check_diagonal_box_membership,
    check_wave_domain_membership,
)
from ecdr.model import OperationalSequentialNumpyCNN
from ecdr.preprocessing import PreprocessingContract


class StreamingReplayEngine:
    """Executes chronological streaming replay on HAPT validation or test streams."""

    def __init__(
        self,
        net: OperationalSequentialNumpyCNN,
        prep: PreprocessingContract,
        domain_type: str = "diagonal_box",
        budget_norm: Union[float, np.ndarray] = 0.001,
        wave_radius: Optional[float] = None,
        use_incremental_reference: bool = True,
        shadow_reference: bool = True,
    ):
        if domain_type not in ("diagonal_box", "wave_domain"):
            raise ValueError(f"Unsupported domain_type: {domain_type}")

        self.net = net
        self.prep = prep
        self.domain_type = domain_type
        self.budget_norm = budget_norm
        self.wave_radius = wave_radius if wave_radius is not None else 0.001
        self.shadow_reference = shadow_reference

        self.cert_mgr = CertificateManager(model_id=net.model_id, prep_id=prep.prep_id)
        self.incremental_engine = (
            IncrementalEngine(net=net, shift=64) if use_incremental_reference else None
        )
        self.instrumentor = CostInstrumentor()

    def run_stream(
        self,
        x_stream: np.ndarray,
        manifest_df: pd.DataFrame,
        measure_baseline: bool = True,
    ) -> Dict[str, dict]:
        """Replay full stream in chronological order with complete wall-clock instrumentation."""
        n_windows = len(x_stream)
        assert len(manifest_df) == n_windows, f"Length mismatch: {n_windows} vs {len(manifest_df)}"

        current_exp = None
        prev_win_idx = -1

        # Complete baseline measurement over full stream workload (preprocessing + reference forward + argmax)
        baseline_latencies = []
        if measure_baseline:
            print("Measuring complete full-reference baseline workload across stream...")
            for idx in range(n_windows):
                tb0 = time.perf_counter()
                norm_w = self.prep.normalize(x_stream[idx])
                res = self.net.forward(norm_w)
                _ = res["pred_class"]
                baseline_latencies.append((time.perf_counter() - tb0) * 1000.0)

        for idx in range(n_windows):
            t_query_start = time.perf_counter()
            row = manifest_df.iloc[idx]
            exp_id = str(row["exp"])
            win_idx = int(row["win_idx"])
            raw_window = x_stream[idx]

            # 1. Normalize window
            norm_window = self.prep.normalize(raw_window)

            # 2. Check recording boundary and sequence continuity
            if exp_id != current_exp:
                self.cert_mgr.reset_recording(exp_id)
                if self.incremental_engine is not None:
                    self.incremental_engine.reset()
                current_exp = exp_id
                prev_win_idx = win_idx
                self.instrumentor.recording_resets += 1
            elif win_idx != prev_win_idx + 1:
                # Sequence gap within same recording -> reset anchor & cache
                self.cert_mgr.reset_recording(exp_id)
                if self.incremental_engine is not None:
                    self.incremental_engine.reset()
                self.instrumentor.sequence_gap_resets += 1

            prev_win_idx = win_idx

            # 3. Check for active certificate and test membership
            cert = self.cert_mgr.get_active_certificate()
            is_hit = False
            fallback_reason = "no_active_cert"
            t_e_sec = 0.0
            t_m_sec = 0.0
            t_v_sec = 0.0
            t_f_sec = 0.0
            t_b_sec = 0.0
            t_shadow_sec = 0.0
            mode = "hit"
            mac_used = 0

            if cert is not None and cert.is_valid_for_recording(exp_id):
                t_v0 = time.perf_counter()
                compat = cert.validate(
                    expected_k=self.net.num_classes,
                    expected_model_id=self.net.model_id,
                    expected_prep_id=self.prep.prep_id,
                    expected_recording_id=exp_id,
                    expected_anchor_window=self.cert_mgr.anchor_window,
                )
                t_v_sec = time.perf_counter() - t_v0

                if compat:
                    t_m0 = time.perf_counter()
                    anchor_w = self.cert_mgr.anchor_window
                    assert anchor_w is not None

                    if cert.domain_type == "diagonal_box":
                        # Direct comparison against exact certified interval endpoints
                        mem_res = check_diagonal_box_membership(
                            z_query=norm_window,
                            z_anchor=anchor_w,
                            budget_norm=cert.budget_norm,
                            domain_box=cert.domain_box,
                        )
                    elif cert.domain_type == "wave_domain":
                        assert cert.wave_box is not None
                        mem_res = check_wave_domain_membership(
                            z_query=norm_window,
                            z_anchor=anchor_w,
                            wave_box=cert.wave_box,
                            shift=64,
                        )
                    else:
                        mem_res = None
                    t_m_sec = time.perf_counter() - t_m0

                    if mem_res is not None and mem_res.is_member:
                        is_hit = True
                        predicted_class = cert.target_class
                        cert.consume_step()
                    else:
                        fallback_reason = mem_res.reason if mem_res is not None else "unsupported_domain"
                else:
                    fallback_reason = "certificate_incompatible_or_nonpositive"

            # 4. Fallback execution if not a hit
            ref_class = None
            if not is_hit:
                t_f0 = time.perf_counter()
                if self.incremental_engine is not None:
                    res_ref = self.incremental_engine.forward_incremental(norm_window, step_idx=idx)
                    mode = str(res_ref["mode"])
                    mac_used = int(res_ref["mac_count"])
                else:
                    res_ref = self.net.forward(norm_window)
                    mode = "full"
                    mac_used = self.instrumentor.full_mac
                t_f_sec = time.perf_counter() - t_f0

                ref_class = res_ref["pred_class"]
                predicted_class = ref_class

                # Attempt certificate build at newly evaluated anchor
                t_b0 = time.perf_counter()
                self.instrumentor.build_attempts += 1
                new_cert = None

                if self.domain_type == "diagonal_box":
                    cert_res = compute_diagonal_box_certificate(
                        net=self.net,
                        z_anchor=norm_window,
                        budget_norm=self.budget_norm,
                        target_class=ref_class,
                    )
                    if cert_res.is_certified:
                        new_cert = Certificate(
                            cert_id=f"diag_{exp_id}_{win_idx}",
                            model_id=self.net.model_id,
                            prep_id=self.prep.prep_id,
                            recording_id=exp_id,
                            anchor_id=win_idx,
                            target_class=ref_class,
                            domain_type="diagonal_box",
                            anchor_state_hash=compute_window_hash(norm_window),
                            domain_box=cert_res.domain_box,
                            budget_norm=np.asarray(self.budget_norm, dtype=np.float64),
                            min_lower_margin=cert_res.min_direct_margin,
                            all_lower_margins=cert_res.direct_lower_margins,
                            horizon=1,
                        )
                elif self.domain_type == "wave_domain":
                    u_center = norm_window[:, 64:]
                    wave_box = Interval.from_box(u_center, self.wave_radius)
                    cert_res = compute_wave_domain_certificate(
                        net=self.net,
                        z_anchor=norm_window,
                        wave_box=wave_box,
                        shift=64,
                        target_class=ref_class,
                    )
                    if cert_res.is_certified:
                        new_cert = Certificate(
                            cert_id=f"wave_{exp_id}_{win_idx}",
                            model_id=self.net.model_id,
                            prep_id=self.prep.prep_id,
                            recording_id=exp_id,
                            anchor_id=win_idx,
                            target_class=ref_class,
                            domain_type="wave_domain",
                            anchor_state_hash=compute_window_hash(norm_window),
                            wave_box=wave_box,
                            min_lower_margin=cert_res.min_direct_margin,
                            all_lower_margins=cert_res.direct_lower_margins,
                            horizon=1,
                        )

                t_b_sec = time.perf_counter() - t_b0

                if new_cert is not None:
                    self.instrumentor.build_successes += 1
                    self.cert_mgr.establish_trusted_anchor(
                        window=norm_window, step_idx=win_idx, recording_id=exp_id, cert=new_cert
                    )
                else:
                    self.instrumentor.build_failures += 1
                    self.cert_mgr.establish_trusted_anchor(
                        window=norm_window, step_idx=win_idx, recording_id=exp_id, cert=None
                    )
            else:
                mode = "hit"
                mac_used = 0
                if self.shadow_reference:
                    t_sh0 = time.perf_counter()
                    res_shadow = self.net.forward(norm_window)
                    t_shadow_sec = time.perf_counter() - t_sh0
                    ref_class = res_shadow["pred_class"]

            # Agreement check (strictly None if shadow reference not checked)
            if is_hit:
                shadow_agreed = (ref_class == predicted_class) if ref_class is not None else None
            else:
                shadow_agreed = None

            t_query_end = time.perf_counter()
            # Operational latency measures the complete query path excluding shadow execution
            t_operational_sec = (t_query_end - t_query_start) - t_shadow_sec

            record = QueryRecord(
                query_idx=idx,
                recording_id=exp_id,
                is_hit=is_hit,
                fallback_reason=fallback_reason if not is_hit else "hit",
                target_class=predicted_class,
                ref_class=ref_class if ref_class is not None else -1,
                shadow_agreement=shadow_agreed,
                t_e_sec=t_e_sec,
                t_m_sec=t_m_sec,
                t_v_sec=t_v_sec,
                t_f_sec=t_f_sec,
                t_b_sec=t_b_sec,
                t_shadow_sec=t_shadow_sec,
                t_operational_sec=t_operational_sec,
                mac_count=mac_used,
                mode=mode,
            )
            self.instrumentor.add_record(record)

        return self.instrumentor.summarize(
            baseline_ref_latencies_ms=baseline_latencies if measure_baseline else None
        )
