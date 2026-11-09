#!/usr/bin/env python3
"""scripts/week4_tables.py — Generates comparison tables, hypothesis ledger, and summary.

Task: W4-035 (Milestone M6)
Hypotheses: H01 through H24

Fulfills W4-035-C01 through C07:
  - Generates output/week4/<run_id>/tables/comparison_matrix.csv
  - Generates output/week4/<run_id>/tables/hypotheses.csv
  - Generates output/week4/<run_id>/tables/h24_conjunction.csv
  - Generates output/week4/<run_id>/tables/summary.md
  - Generates output/week4/<run_id>/tables/statistics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecdr.week4.methods import list_registered_methods
from ecdr.week4.metrics import (
    build_comparison_matrix,
    build_h24_conjunction_matrix,
    build_hypotheses_ledger,
    generate_markdown_summary,
    generate_statistics_summary,
)


def run_tables_generation(
    run_id: str = "week4_initial_20261103",
    base_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
) -> Tuple[Path, Path, Path, Path, Path]:
    base = Path(base_dir) if base_dir else REPO_ROOT
    run_dir = base / f"output/week4/{run_id}"
    tables_dir = Path(output_dir) if output_dir else run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load required measurement artifacts
    replay_json_path = run_dir / "replay_results.json"
    bench_json_path = run_dir / "benchmark_results.json"
    mem_json_path = run_dir / "memory_results.json"

    if not replay_json_path.exists():
        raise FileNotFoundError(f"Missing replay results: {replay_json_path}")
    if not bench_json_path.exists():
        raise FileNotFoundError(f"Missing benchmark results: {bench_json_path}")
    if not mem_json_path.exists():
        raise FileNotFoundError(f"Missing memory results: {mem_json_path}")

    with open(replay_json_path, "r", encoding="utf-8") as f:
        replay_results = json.load(f)
    with open(bench_json_path, "r", encoding="utf-8") as f:
        benchmark_results = json.load(f)
    with open(mem_json_path, "r", encoding="utf-8") as f:
        memory_results = json.load(f)

    raw_methods = list_registered_methods()

    # 1. Comparison Matrix (M00-M16)
    comp_df = build_comparison_matrix(
        replay_results=replay_results,
        benchmark_results=benchmark_results,
        memory_results=memory_results,
        method_registry=raw_methods,
    )
    comp_path = tables_dir / "comparison_matrix.csv"
    comp_df.to_csv(comp_path, index=False)

    # 2. Hypotheses Ledger (H01-H24)
    hyp_df = build_hypotheses_ledger()
    hyp_path = tables_dir / "hypotheses.csv"
    hyp_df.to_csv(hyp_path, index=False)

    # 3. H24 Conjunction Matrix
    h24_df = build_h24_conjunction_matrix(comp_df)
    h24_path = tables_dir / "h24_conjunction.csv"
    h24_df.to_csv(h24_path, index=False)

    # 4. Statistics JSON Summary
    stats_doc = generate_statistics_summary(
        comparison_df=comp_df,
        replay_results=replay_results,
        benchmark_results=benchmark_results,
    )
    stats_path = tables_dir / "statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_doc, f, indent=2)

    # 5. Markdown Synthesis Summary
    md_text = generate_markdown_summary(
        comparison_df=comp_df,
        hypotheses_df=hyp_df,
        h24_df=h24_df,
        stats_doc=stats_doc,
    )
    summary_path = tables_dir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return comp_path, hyp_path, h24_path, summary_path, stats_path


def main():
    parser = argparse.ArgumentParser(description="Week 4 Tables Generator")
    parser.add_argument("--run-id", default="week4_initial_20261103", help="Week 4 Run ID")
    parser.add_argument("--output-dir", default=None, help="Custom output directory")
    args = parser.parse_args()

    c_p, h_p, h24_p, s_p, st_p = run_tables_generation(
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    print("Tables and synthesis generation completed successfully!")
    print(f"  Comparison Matrix: {c_p}")
    print(f"  Hypotheses Ledger: {h_p}")
    print(f"  H24 Conjunction:   {h24_p}")
    print(f"  Summary Report:    {s_p}")
    print(f"  Statistics JSON:   {st_p}")


if __name__ == "__main__":
    main()
