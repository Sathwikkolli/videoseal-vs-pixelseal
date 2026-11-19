#!/usr/bin/env python
"""Build <out>/index.html: a page to watch every watermarked video with its scores.

For each clip (grouped by tier) it shows the side-by-side video (original | models...),
each model's watermarked preview and amplified difference video (|wm - orig| x 20),
the middle-frame PNGs, and quality / robustness numbers. All links are relative, so the
whole <out> folder can be opened in place (Open OnDemand desktop) or copied to a laptop.

Example:
    python scripts/gallery.py --out $VSPS_DATA/out
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

TIERS = ["low", "mid", "high"]
SHOW_ATTACKS = ["none", "compress_crf28", "crop_area75", "downscale_480p", "combined"]

CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e2e0d8;--accent:#2f5bb7}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--fg:#ecebe6;--muted:#a3a29b;--card:#252523;--line:#3a3a37;--accent:#8fb0ff}}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:24px;line-height:1.5}
h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
.muted{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}
.card h3{font-size:16px;margin:0 0 8px}
video{width:100%;border-radius:8px;background:#000}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-top:12px}
.grid figure{margin:0} figcaption{font-size:12px;color:var(--muted)}
img{width:100%;border-radius:6px}
table{border-collapse:collapse;font-size:13px;margin-top:10px;width:100%}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right} th:first-child,td:first-child{text-align:left}
nav a{color:var(--accent);margin-right:12px}
"""


def fmt(v, nd=3):
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return html.escape(str(v))


def rel(p: Path, base: Path) -> str:
    return html.escape(os.path.relpath(p, base).replace(os.sep, "/"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.environ.get("VSPS_DATA", "data"), "out"))
    a = ap.parse_args()
    out = Path(a.out)
    results = [json.loads(f.read_text()) for f in sorted((out / "results").glob("*.json"))]
    if not results:
        print(f"no results in {out / 'results'}", file=sys.stderr)
        return 1

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Watermark viewer</title><style>{CSS}</style></head><body>",
             "<h1>Video Seal vs Pixel Seal: watermarked videos</h1>",
             f"<p class='muted'>{len(results)} clips. Side-by-side order: original, then each model. "
             "Difference videos show |watermarked − original| × 20.</p>",
             "<nav>" + "".join(f"<a href='#{t}'>{t} motion</a>" for t in TIERS) +
             ("<a href='summary/report.md'>report.md</a>" if (out / "summary" / "report.md").exists() else "") + "</nav>"]

    for tier in TIERS:
        rs = sorted([r for r in results if r["tier"] == tier], key=lambda r: r["ti_mean"])
        if not rs:
            continue
        parts.append(f"<h2 id='{tier}'>{tier} motion ({len(rs)} clips)</h2>")
        for r in rs:
            vdir = out / "videos" / r["id"]
            models = list(r["models"])
            order = r.get("side_by_side_order", ["original", *models])
            parts.append(f"<div class='card'><h3>{html.escape(r['id'])}</h3>"
                         f"<div class='muted'>TI {fmt(r['ti_mean'], 1)} · SI {fmt(r['si_mean'], 1)} · "
                         f"{r['width']}×{r['height']} · {r['nframes']} frames · k={r['step_size']} · {html.escape(r['device'])}</div>")
            sbs = vdir / "side_by_side.mp4"
            if sbs.exists():
                parts.append(f"<p class='muted'>{' | '.join(map(html.escape, order))}</p>"
                             f"<video controls preload='metadata' src='{rel(sbs, out)}'></video>")
            parts.append("<div class='grid'>")
            for m in models:
                md = vdir / m
                for fname, cap in (("watermarked.mp4", "watermarked"), ("diff_x20.mp4", "difference ×20")):
                    if (md / fname).exists():
                        parts.append(f"<figure><video controls loop muted preload='metadata' src='{rel(md / fname, out)}'></video>"
                                     f"<figcaption>{html.escape(m)}: {cap}</figcaption></figure>")
                if (md / "frame_diff_x20.png").exists():
                    parts.append(f"<figure><a href='{rel(md / 'frame_watermarked.png', out)}'>"
                                 f"<img loading='lazy' src='{rel(md / 'frame_diff_x20.png', out)}'></a>"
                                 f"<figcaption>{html.escape(m)}: middle-frame difference ×20 (click for full frame)</figcaption></figure>")
            parts.append("</div>")
            head = "".join(f"<th>{html.escape(m)}</th>" for m in models)
            rows = []
            for label, getter, nd in [
                ("PSNR (dB)", lambda m: r["models"][m]["quality"]["psnr"], 2),
                ("SSIM", lambda m: r["models"][m]["quality"]["ssim"], 4),
                ("LPIPS", lambda m: r["models"][m]["quality"]["lpips"], 4),
                ("VMAF", lambda m: r["models"][m]["quality"]["vmaf"], 2),
                ("embed s / video s", lambda m: r["models"][m]["embed"]["model_seconds_per_video_second"], 3),
                ("unwatermarked bit acc", lambda m: r["models"][m]["unwatermarked"]["bit_acc"], 3),
            ] + [(f"bit acc: {atk}", (lambda atk: lambda m: r["models"][m]["attacks"].get(atk, {}).get("bit_acc"))(atk), 3)
                 for atk in SHOW_ATTACKS]:
                rows.append(f"<tr><td>{label}</td>" + "".join(f"<td>{fmt(getter(m), nd)}</td>" for m in models) + "</tr>")
            parts.append(f"<table><tr><th></th>{head}</tr>{''.join(rows)}</table></div>")

    parts.append("</body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
