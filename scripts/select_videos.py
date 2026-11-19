#!/usr/bin/env python
"""Pick N clips per motion tier from candidates.csv using measured TI, not just the label.

  low  <- podcast      : the N lowest-TI podcast clips
  high <- high_motion  : the N highest-TI high-motion clips
  mid  <- gathering    : the N gathering clips whose TI is closest to the midpoint between
                         the low and high tiers, preferring clips strictly between them

With --tiering ti the content labels are ignored: all clips are sorted by TI and cut into
three equal thirds (or the N lowest / N around the geometric midpoint / N highest when there
are more than 3N). Use it when the source labels do not track motion well.

Clip ids listed in <data>/exclude.txt (one per line) are skipped, so off-topic downloads
spotted in candidates.html can be removed. Writes <data>/manifest.csv (one row per selected clip, with its array index for Slurm).

Example:
    python scripts/select_videos.py --data $VSPS_DATA --per-tier 10
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

TIER_OF = {"podcast": "low", "gathering": "mid", "high_motion": "high"}
SOURCES = Path("raw") / "sources.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("VSPS_DATA", "data"))
    ap.add_argument("--per-tier", type=int, default=10)
    ap.add_argument("--metric", default="ti_mean", choices=["ti_mean", "ti"])
    ap.add_argument("--tiering", default="label", choices=["label", "ti"],
                    help="label: tier from content category (default); ti: tier from measured TI only")
    ap.add_argument("--exclude", default=None,
                    help="file with clip ids to drop, one per line (default: <data>/exclude.txt if present)")
    a = ap.parse_args()

    data = Path(a.data)
    with open(data / "candidates.csv", newline="", encoding="utf-8") as f:
        cands = [dict(r, **{a.metric: float(r[a.metric])}) for r in csv.DictReader(f)]
    excl_path = Path(a.exclude) if a.exclude else data / "exclude.txt"
    if excl_path.exists():
        excluded = {ln.strip() for ln in excl_path.read_text(encoding="utf-8").splitlines()
                    if ln.strip() and not ln.startswith("#")}
        before = len(cands)
        cands = [r for r in cands if r["id"] not in excluded]
        print(f"excluded {before - len(cands)} clips listed in {excl_path}")
    src_meta = {}
    if (data / SOURCES).exists():
        with open(data / SOURCES, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                src_meta[Path(r["file"]).stem] = r

    m = a.metric
    if a.tiering == "ti":
        low, mid, high = split_by_ti(cands, m, a.per_tier)
    else:
        low, mid, high = select_by_label(cands, m, a.per_tier)
    return write_manifest(data, src_meta, m, low, mid, high)


def split_by_ti(cands: list[dict], m: str, n: int) -> tuple[list, list, list]:
    rs = sorted(cands, key=lambda r: r[m])
    if len(rs) <= 3 * n:
        k = len(rs) // 3
        extra = len(rs) - 3 * k  # give leftovers to the outer tiers
        a, b = k + (extra > 0), len(rs) - k - (extra > 1)
        return rs[:a], rs[a:b], rs[b:]
    low, high = rs[:n], rs[-n:]
    rest = rs[n:-n]
    centre = math.sqrt(max(low[-1][m], 1e-3) * max(high[0][m], 1e-3))
    mid = sorted(rest, key=lambda r: abs(math.log(max(r[m], 1e-3)) - math.log(centre)))[:n]
    return low, mid, high


def select_by_label(cands: list[dict], m: str, n: int) -> tuple[list, list, list]:
    a_per_tier = n
    by_cat = {c: [r for r in cands if r["category"] == c] for c in TIER_OF}
    for c, rs in by_cat.items():
        if len(rs) < a_per_tier:
            print(f"WARNING: only {len(rs)} '{c}' candidates (need {a_per_tier}); fetch more", file=sys.stderr)
    low = sorted(by_cat["podcast"], key=lambda r: r[m])[: a_per_tier]
    high = sorted(by_cat["high_motion"], key=lambda r: r[m], reverse=True)[: a_per_tier]
    lo_max = max((r[m] for r in low), default=0.0)
    hi_min = min((r[m] for r in high), default=lo_max * 4 or 40.0)
    centre = math.sqrt(max(lo_max, 1e-3) * max(hi_min, 1e-3))  # geometric midpoint
    mids = by_cat["gathering"]
    inside = [r for r in mids if lo_max < r[m] < hi_min]
    outside = [r for r in mids if not (lo_max < r[m] < hi_min)]
    key = lambda r: abs(math.log(max(r[m], 1e-3)) - math.log(centre))  # noqa: E731
    mid = (sorted(inside, key=key) + sorted(outside, key=key))[: a_per_tier]
    return low, mid, high


def write_manifest(data: Path, src_meta: dict, m: str, low: list, mid: list, high: list) -> int:
    rows = []
    for tier, sel in (("low", low), ("mid", mid), ("high", high)):
        for r in sorted(sel, key=lambda r: r[m]):
            meta = src_meta.get(Path(r["raw"]).stem, {})
            rows.append({
                "index": len(rows), "tier": tier, "id": r["id"], "category": r["category"],
                "path": r["path"], "ti_mean": round(float(r["ti_mean"]), 2), "ti": round(float(r["ti"]), 2),
                "si_mean": round(float(r["si_mean"]), 2), "si": round(float(r["si"]), 2),
                "source_page": meta.get("page_url", ""), "author": meta.get("author", ""),
                "license": meta.get("license", ""),
            })

    out = data / "manifest.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'tier':5} {'n':>3} {m + ' range':>18} {'SI mean':>8}")
    for tier in ("low", "mid", "high"):
        t = [r for r in rows if r["tier"] == tier]
        if t:
            vals = [r[m] for r in t]
            print(f"{tier:5} {len(t):3d} {min(vals):8.1f} - {max(vals):6.1f} {sum(r['si_mean'] for r in t)/len(t):8.1f}")
    tiers = {t: [r[m] for r in rows if r["tier"] == t] for t in ("low", "mid", "high")}
    if tiers["mid"] and (min(tiers["mid"]) <= max(tiers["low"], default=-1) or max(tiers["mid"]) >= min(tiers["high"], default=1e9)):
        print("WARNING: TI ranges overlap between tiers; consider fetching more clips", file=sys.stderr)
    print(f"\nwrote {out} ({len(rows)} clips) -> use --array=0-{len(rows) - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
