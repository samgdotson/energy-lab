"""Download EIA-930 six-month BALANCE files for a year (no API key needed).

Always writes a manifest listing which halves were retrieved, so downstream rules can fall
back to other sources (user CSVs, synthetic) when EIA is unreachable or the year predates
EIA-930 (data starts 2015-07-01).
"""

import argparse
import json
from pathlib import Path

import requests

HALVES = ("Jan_Jun", "Jul_Dec")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--url", required=True, help="template with {year} and {half}")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--manifest", required=True)
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"files": [], "failed": {}}
    for half in HALVES:
        url = a.url.format(year=a.year, half=half)
        dest = out_dir / Path(url).name
        try:
            if not dest.exists():
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                # EIA answers missing files (e.g. 2015 Jan-Jun) with an HTML page and HTTP 200
                if not r.content.lstrip().startswith(b'"Balancing Authority"'):
                    raise ValueError("response is not an EIA-930 CSV")
                dest.write_bytes(r.content)
            manifest["files"].append(str(dest))
        except Exception as e:  # network errors, 404 for halves before EIA-930 existed
            manifest["failed"][url] = repr(e)
    Path(a.manifest).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
