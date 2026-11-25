#!/usr/bin/env bash
set -euo pipefail
report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$report_dir/build_sources.py"
for report_stem in ECDR_Weeks_1_2_3_Theory Week_1_Project_Brief Week_2_Data_Foundations Week_3_Mathematical_Foundations ECDR_Mathematics_Explained ECDR_Data_Analysis ECDR_CNN_Model_Analysis; do
  tectonic --keep-logs --outdir "$report_dir" "$report_dir/$report_stem.tex"
done
