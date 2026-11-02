"""instrumentation.py — Complete cost instrumentation and accounting.

Authoritative source: 02_MODEL/02_ECDR_CNN_Model_Analysis_VI.tex (Sections 11.2, 12)
and 07_G3_DOMAIN_CONTRACT_VI.md (Section 7).

Tracks per-query and aggregate execution costs:
- T_e: event / domain candidate selection duration (unmeasured = 0.0)
- T_m: conservative membership check duration
- T_v: certificate gate / identity validation duration
- T_f: full reference forward duration (on fallback)
- T_b: certificate build duration (when attempted)
- T_shadow: shadow reference forward duration (tracked separately from operational latency)
- T_operational: pure operational query latency excluding shadow execution
- Shadow reference agreement: verified on accepted queries, None if unchecked
- Theoretical MAC vs measured wall-clock comparisons
"""

from dataclasses import dataclass, field
import json
import time
from typing import Dict, List, Optional
import numpy as np


@dataclass
class QueryRecord:
    query_idx: int
    recording_id: str
    is_hit: bool
    fallback_reason: str
    target_class: int
    ref_class: int
    shadow_agreement: Optional[bool]  # None if shadow reference not checked
    t_e_sec: float = 0.0
    t_m_sec: float = 0.0
    t_v_sec: float = 0.0
    t_f_sec: float = 0.0
    t_b_sec: float = 0.0
    t_shadow_sec: float = 0.0
    t_operational_sec: float = 0.0
    mac_count: int = 0
    mode: str = "hit"  # "hit", "incremental", "full"


class CostInstrumentor:
    """Instrumentor collecting fine-grained latency and event metrics."""

    def __init__(self, full_mac: int = 2343936, incremental_mac: int = 1241856):
        self.full_mac = full_mac
        self.incremental_mac = incremental_mac

        self.records: List[QueryRecord] = []
        self.total_queries = 0
        self.hits = 0
        self.misses = 0
        self.build_attempts = 0
        self.build_successes = 0
        self.build_failures = 0
        self.recording_resets = 0
        self.sequence_gap_resets = 0

    def add_record(self, record: QueryRecord) -> None:
        self.records.append(record)
        self.total_queries += 1
        if record.is_hit:
            self.hits += 1
        else:
            self.misses += 1

    def summarize(self, baseline_ref_latencies_ms: Optional[List[float]] = None) -> Dict[str, dict]:
        """Compute aggregate statistics, percentiles, and cost accounting."""
        n = self.total_queries
        if n == 0:
            return {"status": "no_queries_recorded"}

        t_e = [r.t_e_sec * 1000.0 for r in self.records]
        t_m = [r.t_m_sec * 1000.0 for r in self.records]
        t_v = [r.t_v_sec * 1000.0 for r in self.records]
        t_f = [r.t_f_sec * 1000.0 for r in self.records if r.t_f_sec > 0]
        t_b = [r.t_b_sec * 1000.0 for r in self.records if r.t_b_sec > 0]
        t_shadow = [r.t_shadow_sec * 1000.0 for r in self.records if r.t_shadow_sec > 0]
        t_op = [r.t_operational_sec * 1000.0 for r in self.records]
        macs = [r.mac_count for r in self.records]

        # Shadow reference agreement: strictly count only where checked
        checked_agreements = [
            r.shadow_agreement for r in self.records if r.is_hit and r.shadow_agreement is not None
        ]
        checked_count = len(checked_agreements)
        if checked_count > 0:
            agreed_count = sum(1 for a in checked_agreements if a is True)
            agreement_rate = float(agreed_count / checked_count)
            discrepancies = checked_count - agreed_count
        else:
            agreement_rate = None
            agreed_count = 0
            discrepancies = 0

        hit_rate = self.hits / float(n)
        mean_mac = float(np.mean(macs))
        mac_saving_pct = (1.0 - mean_mac / float(self.full_mac)) * 100.0

        def stats(arr: list) -> dict:
            if not arr:
                return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
            return {
                "count": len(arr),
                "mean_ms": float(np.mean(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
            }

        summary = {
            "query_counts": {
                "total_queries": n,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_pct": hit_rate * 100.0,
                "build_attempts": self.build_attempts,
                "build_successes": self.build_successes,
                "build_failures": self.build_failures,
                "recording_resets": self.recording_resets,
                "sequence_gap_resets": self.sequence_gap_resets,
            },
            "shadow_agreement": {
                "total_hits": self.hits,
                "checked_accepted_queries": checked_count,
                "agreed_queries": agreed_count,
                "agreement_rate": agreement_rate,
                "discrepancies": discrepancies,
            },
            "latencies_ms": {
                "t_e_candidate_selection": stats(t_e),
                "t_m_membership": stats(t_m),
                "t_v_gate_validation": stats(t_v),
                "t_f_full_forward": stats(t_f),
                "t_b_certificate_build": stats(t_b),
                "t_shadow_execution": stats(t_shadow),
                "operational_query_latency": stats(t_op),
            },
            "computation_mac": {
                "full_reference_mac_per_window": self.full_mac,
                "incremental_mac_per_window": self.incremental_mac,
                "average_mac_per_window": mean_mac,
                "theoretical_mac_saving_pct": mac_saving_pct,
            },
        }

        if baseline_ref_latencies_ms:
            baseline_mean = float(np.mean(baseline_ref_latencies_ms))
            op_mean = float(np.mean(t_op))
            speedup = baseline_mean / max(1e-9, op_mean)
            summary["empirical_speedup"] = {
                "measured_baseline_mean_ms": baseline_mean,
                "measured_operational_mean_ms": op_mean,
                "speedup_factor": speedup,
            }

        return summary
