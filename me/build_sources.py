#!/usr/bin/env python3
"""Extract standalone weekly LaTeX reports and sync canonical extracts.

Canonical Week 1 authority is owned by 00_OVERVIEW/02_Week_1_Project_Brief.tex.
This script extracts the canonical Week 1 body to 00_OVERVIEW/week1/brief_body.tex
for combined master / compat wrappers. Week 2 has its own canonical source in
01_DATA; its compatibility wrapper must never be regenerated from the old master.
Week 3 wrappers now resolve to canonical 03_MATH sources; the old combined master remains historical.
"""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent

BEGIN_CANONICAL_WEEK1 = "% BEGIN CANONICAL WEEK1 BODY\n"
END_CANONICAL_WEEK1 = "% END CANONICAL WEEK1 BODY"
GENERATED_WEEK1_PREFIX = "% GENERATED from 00_OVERVIEW/02_Week_1_Project_Brief.tex. Do not edit.\n"


def extract_canonical_week1_body(text: str) -> str:
    """Extract Week 1 body from canonical source bounded by exact markers.

    Validates exactly one pair of markers in forward order. Fails clearly if missing
    or ambiguous; never falls back to regenerating Week 1 from legacy master.
    """
    begin_count = text.count(BEGIN_CANONICAL_WEEK1)
    end_count = text.count(END_CANONICAL_WEEK1)
    if begin_count != 1 or end_count != 1:
        raise ValueError(
            f"Canonical Week 1 markers invalid: found {begin_count} begin and "
            f"{end_count} end markers (expected exactly 1 pair)."
        )
    parts = text.split(BEGIN_CANONICAL_WEEK1, 1)
    if END_CANONICAL_WEEK1 not in parts[1]:
        raise ValueError("Canonical Week 1 markers out of order.")
    return parts[1].split(END_CANONICAL_WEEK1, 1)[0]


def sync_generated_week1_body(canonical_path: Path, output_path: Path) -> bool:
    """Sync derivative week1/brief_body.tex from authoritative canonical source."""
    if not canonical_path.exists():
        raise FileNotFoundError(f"Authoritative canonical source missing: {canonical_path}")
    text = canonical_path.read_text(encoding="utf-8")
    body = extract_canonical_week1_body(text)
    content = GENERATED_WEEK1_PREFIX + body
    if not output_path.exists() or output_path.read_text(encoding="utf-8") != content:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"Updated {output_path} from canonical Week 1 markers.")
        return True
    return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract standalone weekly LaTeX reports and sync canonical extracts."
    )
    parser.add_argument(
        "--week2-only",
        action="store_true",
        help="Sync only the Week 2 compatibility wrapper from its canonical 01_DATA source; preserve other weeks.",
    )
    parser.add_argument(
        "--week1-only",
        action="store_true",
        help="Sync generated Week 1 body extract from canonical authority and return before legacy Week 2/3 extraction.",
    )
    parser.add_argument('--week3-only', action='store_true',
                        help='Sync only Week 3 compatibility wrappers from canonical 03_MATH sources.')
    return parser.parse_args(argv)


def sync_week2_wrapper() -> None:
    canonical = ROOT.parent / "01_DATA" / "03_Week_2_Data_Foundations.tex"
    if not canonical.is_file():
        raise FileNotFoundError(f"Authoritative Week 2 source missing: {canonical}")
    wrapper = ROOT / "Week_2_Data_Foundations.tex"
    content = (
        "% GENERATED compatibility wrapper. Edit 01_DATA/03_Week_2_Data_Foundations.tex.\n"
        "\\IfFileExists{01_DATA/03_Week_2_Data_Foundations.tex}{%\n"
        "  \\input{01_DATA/03_Week_2_Data_Foundations.tex}%\n"
        "}{%\n"
        "  \\input{../01_DATA/03_Week_2_Data_Foundations.tex}%\n"
        "}\n"
    )
    if not wrapper.exists() or wrapper.read_text(encoding="utf-8") != content:
        wrapper.write_text(content, encoding="utf-8")
        print(f"Updated {wrapper} to reference canonical Week 2 source.")


def sync_week3_wrappers() -> None:
    for wrapper_name, canonical_name in [
        ('Week_3_Mathematical_Foundations', '03_Week_3_Mathematical_Foundations'),
        ('ECDR_Mathematics_Explained', '04_ECDR_Mathematics_Explained'),
    ]:
        relative = f'03_MATH/{canonical_name}.tex'
        canonical = ROOT.parent / relative
        if not canonical.is_file():
            raise FileNotFoundError(f'Authoritative Week 3 source missing: {canonical}')
        content = (
            f'% GENERATED compatibility wrapper. Edit {relative}.\n'
            f'\\IfFileExists{{{relative}}}{{%\n'
            f'  \\input{{{relative}}}%\n'
            '}{%\n'
            f'  \\input{{../{relative}}}%\n'
            '}\n'
        )
        wrapper = ROOT / f'{wrapper_name}.tex'
        if not wrapper.exists() or wrapper.read_text(encoding='utf-8') != content:
            wrapper.write_text(content, encoding='utf-8')
            print(f'Updated {wrapper} from canonical Week 3 source.')


def main(argv=None) -> None:
    args = parse_args(argv)
    if sum([args.week1_only, args.week2_only, args.week3_only]) > 1:
        raise ValueError('Choose only one week-specific sync option')
    if args.week3_only:
        sync_week3_wrappers()
        return
    if args.week2_only:
        sync_week2_wrapper()
        return
    canonical_file = ROOT.parent / '00_OVERVIEW' / '02_Week_1_Project_Brief.tex'
    generated_file = ROOT.parent / '00_OVERVIEW' / 'week1' / 'brief_body.tex'
    sync_generated_week1_body(canonical_file, generated_file)
    if args.week1_only:
        return
    sync_week2_wrapper()
    sync_week3_wrappers()


if __name__ == '__main__':
    main()
