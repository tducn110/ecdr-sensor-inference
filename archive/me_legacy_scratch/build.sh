#!/usr/bin/env bash
set -euo pipefail
report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
research_root="$(cd -- "$report_dir/.." && pwd)"
overview_dir="$research_root/00_OVERVIEW"

# Honor existing externally provided TECTONIC_BIN and cache; fallback to resource path or system
if [[ -z "${TECTONIC_BIN:-}" ]]; then
  if [[ -x "/usr/lib/chatgpt/resources/tectonic/tectonic" ]]; then
    TECTONIC_BIN="/usr/lib/chatgpt/resources/tectonic/tectonic"
  elif command -v tectonic >/dev/null 2>&1; then
    TECTONIC_BIN="$(command -v tectonic)"
  else
    TECTONIC_BIN="tectonic"
  fi
fi

if [[ -n "${TECTONIC_CACHE_DIR:-}" ]]; then
  export TECTONIC_CACHE_DIR
elif [[ -n "${TECTONIC_CACHE:-}" ]]; then
  export TECTONIC_CACHE_DIR="$TECTONIC_CACHE"
elif [[ -d "/usr/lib/chatgpt/resources/tectonic/cache" ]]; then
  export TECTONIC_CACHE_DIR="/usr/lib/chatgpt/resources/tectonic/cache"
fi

python_bin="/home/pro/Downloads/research/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

"$python_bin" "$report_dir/build_sources.py"

# Compile canonical Week 1 directly to 00_OVERVIEW (not rewrite it)
"$TECTONIC_BIN" --keep-logs --outdir "$overview_dir" "$overview_dir/02_Week_1_Project_Brief.tex"

# Legacy compilation loop: skip Week_1_Project_Brief; all other stems unchanged
for report_stem in ECDR_Weeks_1_2_3_Theory Week_1_Project_Brief Week_2_Data_Foundations Week_3_Mathematical_Foundations ECDR_Mathematics_Explained ECDR_Data_Analysis ECDR_CNN_Model_Analysis; do
  if [[ "$report_stem" == "Week_1_Project_Brief" ]]; then
    continue
  fi
  "$TECTONIC_BIN" --keep-logs --outdir "$report_dir" "$report_dir/$report_stem.tex"
done
