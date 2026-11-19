#!/usr/bin/env python
"""Collect per-video JSON results into tables, statistics and plots.

Outputs in <out>/summary/:
  robustness_long.csv   one row per video x model x attack
  quality_long.csv      one row per video x model (PSNR, SSIM, LPIPS, VMAF, embed time)
  robustness_summary.csv  mean bit accuracy / log10 p with 95% bootstrap CI, per tier x model x attack
  quality_summary.csv     same for quality and timing
  paired_tests.csv      Wilcoxon signed-rank, model A vs model B on the same videos, per tier x attack
  plots/*.png           bit accuracy vs attack strength, per tier
  report.md             the main tables in markdown

Example:
    python scripts/aggregate.py --out $VSPS_DATA/out
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TIERS = ["low", "mid", "high"]


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    out = []
    for k, g in df.groupby(keys, sort=False):
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        row["n_videos"] = g["id"].nunique()
        for c in cols:
            v = g[c].astype(float).to_numpy()
            lo, hi = boot_ci(v)
            row[f"{c}_mean"], row[f"{c}_ci_lo"], row[f"{c}_ci_hi"] = float(np.nanmean(v)), lo, hi
        out.append(row)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.environ.get("VSPS_DATA", "data"), "out"))
    a = ap.parse_args()

    out = Path(a.out)
    files = sorted((out / "results").glob("*.json"))
    if not files:
        print(f"no results in {out / 'results'}", file=sys.stderr)
        return 1
    rob, qual = [], []
    for f in files:
        r = json.loads(f.read_text())
        base = {k: r[k] for k in ("id", "tier", "ti_mean", "si_mean")}
        for model, m in r["models"].items():
            e = m["embed"]
            qual.append({**base, "model": model, **m["quality"],
                         "embed_s_per_video_s": e["model_seconds_per_video_second"],
                         "embed_wall_s_per_video_s": e["wall_seconds_per_video_second"],
                         "embed_fps": e["model_fps"], "peak_gpu_mem_gb": e.get("peak_gpu_mem_gb"),
                         "unwm_bit_acc": m["unwatermarked"]["bit_acc"],
                         "unwm_false_positive": m["unwatermarked"]["detected_fpr1e-6"], "device": r["device"]})
            for atk, s in m["attacks"].items():
                rob.append({**base, "model": model, "attack": atk, "family": s["family"], "strength": s["strength"],
                            "bit_acc": s["bit_acc"], "log10_p": s["log10_p"],
                            "detected": s["detected_fpr1e-6"]})
    rob, qual = pd.DataFrame(rob), pd.DataFrame(qual)
    for d in (rob, qual):
        d["tier"] = pd.Categorical(d["tier"], TIERS, ordered=True)
    attack_order = list(dict.fromkeys(rob["attack"]))
    rob["attack"] = pd.Categorical(rob["attack"], attack_order, ordered=True)
    rob = rob.sort_values(["tier", "attack", "model", "id"])
    qual = qual.sort_values(["tier", "model", "id"])

    sdir = out / "summary"
    (sdir / "plots").mkdir(parents=True, exist_ok=True)
    rob.to_csv(sdir / "robustness_long.csv", index=False)
    qual.to_csv(sdir / "quality_long.csv", index=False)

    rsum = summarize(rob, ["tier", "attack", "model"], ["bit_acc", "log10_p", "detected"])
    qcols = ["psnr", "ssim", "lpips", "vmaf", "embed_s_per_video_s", "embed_wall_s_per_video_s", "unwm_bit_acc"]
    qual["vmaf"] = pd.to_numeric(qual["vmaf"], errors="coerce")
    qsum = summarize(qual, ["tier", "model"], qcols)
    rsum.to_csv(sdir / "robustness_summary.csv", index=False)
    qsum.to_csv(sdir / "quality_summary.csv", index=False)

    # paired comparison on identical videos
    from scipy.stats import wilcoxon
    tests = []
    models = list(dict.fromkeys(rob["model"]))
    for (ma, mb) in itertools.combinations(models, 2):
        for (tier, atk), g in rob.groupby(["tier", "attack"], observed=True):
            p = g.pivot_table(index="id", columns="model", values="bit_acc", observed=True)
            if ma not in p or mb not in p:
                continue
            p = p[[ma, mb]].dropna()
            diff = (p[ma] - p[mb]).to_numpy()
            try:
                pval = float(wilcoxon(p[ma], p[mb]).pvalue) if np.any(diff != 0) and len(p) >= 5 else float("nan")
            except ValueError:
                pval = float("nan")
            tests.append({"tier": tier, "attack": atk, "model_a": ma, "model_b": mb, "n": len(p),
                          "mean_a": p[ma].mean(), "mean_b": p[mb].mean(), "mean_diff_a_minus_b": diff.mean(),
                          "wilcoxon_p": pval})
    tests = pd.DataFrame(tests)
    tests.to_csv(sdir / "paired_tests.csv", index=False)

    # plots: bit accuracy vs strength per family, one figure per tier
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [f for f in ["compress", "crop", "downscale", "combined"] if f in set(rob["family"])]
    for tier in TIERS:
        g = rob[rob["tier"] == tier]
        if g.empty:
            continue
        fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 3.4), sharey=True, squeeze=False)
        for ax, fam in zip(axes[0], fams):
            gf = g[g["family"] == fam]
            for model in models:
                gm = gf[gf["model"] == model].groupby("strength", sort=False, observed=True)["bit_acc"]
                means = gm.mean()
                cis = [boot_ci(v.to_numpy()) for _, v in gm]
                x = np.arange(len(means))
                ax.errorbar(x, means.values, yerr=[means.values - [c[0] for c in cis], [c[1] for c in cis] - means.values],
                            marker="o", capsize=3, label=model)
                ax.set_xticks(x, means.index, rotation=20, fontsize=8)
            ax.set_title(fam)
            ax.set_ylim(0.45, 1.02)
            ax.axhline(0.5, color="grey", lw=0.6, ls=":")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("bit accuracy")
        axes[0][-1].legend(fontsize=8)
        fig.suptitle(f"{tier} motion (n={g['id'].nunique()} videos)")
        fig.tight_layout()
        fig.savefig(sdir / "plots" / f"robustness_{tier}.png", dpi=150)
        plt.close(fig)

    # markdown report
    lines = ["# Video Seal vs Pixel Seal\n", f"{rob['id'].nunique()} videos, device: {', '.join(qual['device'].unique())}\n",
             "## Quality and embedding time (mean per tier)\n"]
    qt = qsum[["tier", "model", "n_videos", "psnr_mean", "ssim_mean", "lpips_mean", "vmaf_mean",
               "embed_s_per_video_s_mean", "unwm_bit_acc_mean"]].round(4)
    lines += [qt.to_markdown(index=False), "\n## Bit accuracy (mean [95% CI])\n"]
    rsum["cell"] = rsum.apply(lambda r: f"{r.bit_acc_mean:.3f} [{r.bit_acc_ci_lo:.3f}, {r.bit_acc_ci_hi:.3f}]", axis=1)
    piv = rsum.pivot_table(index=["tier", "attack"], columns="model", values="cell", aggfunc="first", observed=True)
    lines += [piv.reset_index().to_markdown(index=False), "\n## Paired Wilcoxon tests on bit accuracy\n",
              tests.round(4).to_markdown(index=False) if not tests.empty else "(not enough data)", ""]
    (sdir / "report.md").write_text("\n".join(lines))
    print((sdir / "report.md").read_text())
    print(f"\nsummary written to {sdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
