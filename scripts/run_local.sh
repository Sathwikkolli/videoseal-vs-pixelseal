#!/bin/bash
# Run every manifest row sequentially on this machine (no Slurm). Rows whose result JSON
# already exists are skipped, so it is safe to re-run after adding clips to the manifest.
#
#   bash scripts/run_local.sh data/manifest.csv data/out [extra run_one.py args...]
#
# CPU-friendly example (what the local laptop run used):
#   bash scripts/run_local.sh data/manifest.csv data/out --device cpu --extract-seconds 3 \
#        --attacks compress_crf28 crop_area75 downscale_480p combined
set -u
manifest=${1:?manifest.csv}; out=${2:?output dir}; shift 2
py=${PYTHON:-python}
n=$(($(wc -l < "$manifest") - 1))
mkdir -p "$out/logs"
for ((i = 0; i < n; i++)); do
    id=$(awk -F, -v r=$((i + 2)) 'NR == r {print $3}' "$manifest")
    if [ -f "$out/results/$id.json" ]; then
        echo "[$((i + 1))/$n] $id done, skipping"
        continue
    fi
    echo "[$((i + 1))/$n] $id"
    "$py" -W ignore "$(dirname "$0")/run_one.py" --manifest "$manifest" --index "$i" --out "$out" "$@" \
        > "$out/logs/$id.log" 2>&1 || echo "    FAILED (see $out/logs/$id.log)"
done
echo "all done"
