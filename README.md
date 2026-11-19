# Video Seal vs Pixel Seal

A benchmark that compares two open video watermarking models from Meta,
[Video Seal 1.0](https://arxiv.org/abs/2412.09492) and
[Pixel Seal](https://arxiv.org/abs/2512.16874), on videos with low, mid and high
amounts of motion. Each video goes through three everyday attacks plus one combined
attack. The jobs run as Slurm jobs on UMich **Great Lakes**.

| Tier | Content | How clips are chosen |
|---|---|---|
| low | podcasts, static camera, 1–2 people talking | the 10 lowest-TI podcast clips |
| mid | a speaker at a public gathering, a moving crowd, slow pans | the 10 gathering clips whose TI falls between the low and high tiers |
| high | sports, concerts, dance, fireworks, handheld camera | the 10 highest-TI high-motion clips |

Clips are placed in tiers by **measured motion**, not by label alone. Motion is the ITU-T
P.910 temporal information, TI = std(Y<sub>t</sub> − Y<sub>t−1</sub>). Spatial
information (SI, texture) is reported too.

## What gets measured

| | |
|---|---|
| Attacks (real ffmpeg) | H.264 re-encode at CRF 23 / 28 / 33 · centre crop keeping 90 / 75 / 50 % of the area · downscale to 720p / 480p / 360p · combined (crop 75 % → 480p → CRF 28) |
| Robustness | bit accuracy, log10 p-value, detection at FPR 10⁻⁶ |
| Negative control | decoding run on the *unwatermarked* clip (should be about 50 % bit accuracy) |
| Quality | PSNR (all frames), SSIM and LPIPS (every 10th frame), VMAF |
| Speed | embedding time (model forward only, and wall clock), per second of video, plus peak GPU memory |
| Statistics | mean with 95 % bootstrap CI per tier; paired Wilcoxon test between the models on the same videos |

To make the comparison fair:
- Both models carry 256 bits.
- Each video gets one message, shared by both models.
- Both use the same watermark step size, `k = 4`. Pixel Seal's own default is 8; add `pixelseal@k8` to `VSPS_MODELS` to test that too.
- All tasks run on one GPU type.

## Repository layout

```
scripts/fetch_videos.py    download candidate clips (Wikimedia Commons / Pixabay / Pexels) + attribution CSV
scripts/prepare.py         normalize to 1080p30, 10 s; compute SI/TI -> candidates.csv + candidates.html
scripts/select_videos.py   choose 10 clips per tier by TI -> manifest.csv
scripts/run_one.py         one clip: embed, quality, attacks, extract, viewer files -> results/<id>.json
scripts/aggregate.py       tables, CIs, paired tests, plots, report.md
scripts/gallery.py         index.html to watch every watermarked video
src/vsbench/               video I/O, metrics, attacks, model wrappers
slurm/                     config.sh, setup_env.sh, prepare / run_array / aggregate .sbatch
tests/                     model-free unit tests
```

## Running on Great Lakes

All paths default to `slurm/config.sh`:
- account: `hafiz1`
- data: `/scratch/hafiz1_root/hafiz1/$USER/vsps`
- conda env: `/scratch/hafiz1_root/hafiz1/$USER/envs/vsps` (env and pip/conda/torch caches are kept on scratch, outside the 80 GB home quota)

Override any of them with environment variables before running `sbatch`.

### Fast path: use the released clips (no fetching or preparing)

The 38 prepared clips and their manifest (tiers by measured TI) are attached to the
[`clips-v1` release](https://github.com/Sathwikkolli/videoseal-vs-pixelseal/releases/tag/clips-v1).

```bash
git clone https://github.com/Sathwikkolli/videoseal-vs-pixelseal.git ~/videoseal-vs-pixelseal
cd ~/videoseal-vs-pixelseal && bash slurm/setup_env.sh            # once, ~15-25 min
source slurm/config.sh && mkdir -p "$VSPS_DATA" logs
curl -L -o "$VSPS_DATA/vsps_clips.tar" https://github.com/Sathwikkolli/videoseal-vs-pixelseal/releases/download/clips-v1/vsps_clips.tar
tar -xf "$VSPS_DATA/vsps_clips.tar" -C "$VSPS_DATA" --strip-components=1
sed -i "s#,norm/#,$VSPS_DATA/norm/#" "$VSPS_DATA/manifest.csv"
sbatch --array=0-37 slurm/run_array.sbatch                         # one GPU job per clip
# when squeue -u $USER shows no vsps_run jobs left:
sbatch slurm/aggregate.sbatch                                      # tables, plots, report.md, index.html
```

```bash
# 0. once, on a login node
git clone https://github.com/Sathwikkolli/videoseal-vs-pixelseal.git ~/videoseal-vs-pixelseal
bash ~/videoseal-vs-pixelseal/slurm/setup_env.sh

# 1. get candidate clips (login node). Wikimedia Commons needs no key:
source ~/videoseal-vs-pixelseal/slurm/config.sh && vsps_activate
export VSPS_UA="vsbench-video-fetcher/0.1 (contact: <your email>)"   # Wikimedia asks for a contact
python ~/videoseal-vs-pixelseal/scripts/fetch_videos.py --data $VSPS_DATA --source commons --per-category 30
#    optional, better search relevance: free Pixabay key from https://pixabay.com/api/docs/
#    export PIXABAY_API_KEY=... and use --source pixabay commons

# 2. normalize, measure TI/SI, select 10 per tier (CPU job)
cd ~/videoseal-vs-pixelseal && mkdir -p logs
sbatch slurm/prepare.sbatch

# 3. review $VSPS_DATA/candidates.html (thumbnail + TI per clip). Put the ids of
#    off-topic clips in $VSPS_DATA/exclude.txt, one per line, and re-select:
python scripts/select_videos.py --data $VSPS_DATA --per-tier 10
#    then run the 30 clips on GPUs
jid=$(sbatch --parsable slurm/run_array.sbatch)

# 4. tables, plots and the video viewer once the array is done
sbatch --dependency=afterany:$jid slurm/aggregate.sbatch
```

You can also put your own clips in `$VSPS_DATA/raw/{podcast,gathering,high_motion}/` before
step 2; `prepare.py` picks up everything in those folders.

Useful variations:

```bash
VSPS_MODELS="videoseal pixelseal pixelseal@k8" sbatch slurm/run_array.sbatch   # also Pixel Seal at its default k=8
sbatch --partition=spgpu slurm/run_array.sbatch                                # A40s instead of V100s
sbatch --array=7 slurm/run_array.sbatch                                         # rerun one clip
```

## Watching the watermarked videos

Each clip gets a folder `$VSPS_OUT/videos/<id>/` containing:

- `original.mp4`
- `side_by_side.mp4`: original | Video Seal | Pixel Seal
- `<model>/watermarked.mp4`
- `<model>/diff_x20.mp4`: |watermarked − original| × 20, which shows where the watermark is
- `<model>/frame_*.png`: the middle frame as original, watermarked and difference

`$VSPS_OUT/index.html` puts all of them on one page, together with each clip's scores.

There are three ways to open it:

1. **Open OnDemand → Jupyter** (easiest; videos play in your own browser). Before starting
   the session, run `ln -s $VSPS_OUT ~/vsps_out` once on a login node. Then start a Jupyter
   session at https://greatlakes.arc-ts.umich.edu, browse to `vsps_out/`, right-click
   `index.html` and choose **Open in New Browser Tab**.
2. **Copy the files to your laptop.** The preview videos are small.

   ```bash
   rsync -av --exclude 'attacked_*' <uniqname>@greatlakes-xfer.arc-ts.umich.edu:/scratch/hafiz1_root/hafiz1/<uniqname>/vsps/out/ ./vsps_out/
   ```

   Then open `vsps_out/index.html`.
3. **Open OnDemand → Remote Desktop.** Open the `.mp4` files directly in a video player there.

The preview files are re-encoded at CRF 18 so browsers can play them. All metrics are
computed on lossless copies, not on these previews.

## Local test

```bash
python -m unittest discover -s tests -v
```

The unit tests need ffmpeg but no models or GPU.

### Full run on a laptop (no Slurm, CPU)

```bash
python scripts/prepare.py --data data --duration 5 --workers 2
python scripts/select_videos.py --data data --per-tier 10
bash scripts/run_local.sh data/manifest.csv data/out --device cpu --extract-seconds 3 \
     --attacks compress_crf28 crop_area75 downscale_480p combined
python scripts/aggregate.py --out data/out && python scripts/gallery.py --out data/out
```

This reduced setup uses 5 s clips, decodes from the first 3 s (as in the paper's Table 4) and
runs one strength per attack family. On CPU it takes roughly 10-15 min per clip. CPU
embedding times are not comparable with GPU numbers.

## Notes

- Clips come from [Wikimedia Commons](https://commons.wikimedia.org) (CC / public-domain;
  the license of each file is recorded) and optionally [Pixabay](https://pixabay.com/service/license-summary/)
  or [Pexels](https://www.pexels.com/license/). Pexels has paused new API keys. Authors,
  page URLs and licenses are recorded in `raw/sources.csv` and copied into `manifest.csv`.
- Commons search is title-based and noisy, which is why the review step exists. Long
  videos are cut 20 % in (`prepare.py --start auto`) to skip intros and title cards.
- SA-V is deliberately not used, because Video Seal was trained on it.
- Video Seal is pinned to commit `f12875a` (see `slurm/setup_env.sh`).
