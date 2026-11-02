#!/usr/bin/env python3
"""01_fetch_hapt.py — Download HAPT (UCI id=341), record SHA-256 + provenance."""
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = ("https://archive.ics.uci.edu/static/public/341/"
       "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip")
ZIP = DATA / "hapt_uci341.zip"
MANIFEST = DATA / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not ZIP.exists() or ZIP.stat().st_size < 1_000_000:
        print(f"downloading {URL}")
        tmp = ZIP.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)  # noqa: S310 (fixed UCI URL)
        tmp.rename(ZIP)
    size = ZIP.stat().st_size
    digest = sha256(ZIP)
    with zipfile.ZipFile(ZIP) as zf:
        names = zf.namelist()
        zf.extractall(DATA / "hapt")
    # normalize: UCI zip may nest under a folder
    inner = [p for p in (DATA / "hapt").iterdir() if p.is_dir()]
    if len(inner) == 1 and not (DATA / "hapt" / "RawData").exists():
        for p in inner[0].iterdir():
            shutil.move(str(p), DATA / "hapt" / p.name)
        inner[0].rmdir()
    manifest = {
        "source_url": URL,
        "accessed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zip_path": str(ZIP.relative_to(ROOT)),
        "zip_size_bytes": size,
        "zip_sha256": digest,
        "zip_member_count": len(names),
        "extract_dir": "data/hapt",
        "download_seconds": round(time.time() - t0, 1),
        "doi": "10.24432/C54G7M",
        "license": "CC BY 4.0",
        "note": ("10929 instances on UCI page = pre-windowed 561-feature records, "
                 "NOT raw sensor samples; raw row counts must be read from files"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
