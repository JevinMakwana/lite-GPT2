#!/usr/bin/env bash
# Download and prepare all data needed by train_gpt2.py:
#   1. FineWeb-Edu sample-10BT  -> edu_fineweb10B/  (100 shards of 100M GPT-2 tokens, ~20 GB)
#   2. HellaSwag validation set -> hellaswag/hellaswag_val.jsonl (~11 MB)
#
# Usage (run from anywhere; paths are relative to this script's folder):
#   bash setup_data.sh
#
# Optional environment variables:
#   HF_HOME=/big/disk/hf_cache   where HuggingFace stores the raw download + arrow cache
#                                (default ~/.cache/huggingface; needs ~70-80 GB)
#   STREAM=1                     stream FineWeb-Edu from HuggingFace instead of caching it
#                                locally (no ~70-80 GB HF cache; an interrupted run restarts)
#   SKIP_PIP=1                   don't pip-install dependencies
#   CLEAN_HF_CACHE=1             delete the raw FineWeb-Edu HF cache after tokenizing
#
# Safe to re-run: finished steps are skipped. If tokenizing is interrupted,
# delete edu_fineweb10B/ and run again (the HF download itself resumes from cache).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
SHARD_DIR="edu_fineweb10B"
EXPECTED_SHARDS=100
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

echo "== Working directory: $SCRIPT_DIR"
echo "== Python: $($PYTHON --version 2>&1) ($(command -v $PYTHON))"
echo "== HuggingFace cache: $HF_CACHE"
STREAM="${STREAM:-0}"
export STREAM   # read by fineweb.py
if [[ "$STREAM" == "1" ]]; then
    echo "== STREAM=1: FineWeb-Edu will be streamed, not saved to the HF cache"
fi

# --- 1. dependencies ----------------------------------------------------------
if [[ "${SKIP_PIP:-0}" != "1" ]]; then
    echo "== Installing Python dependencies"
    $PYTHON -m pip install --upgrade datasets tiktoken numpy tqdm
fi

# --- 2. disk space check --------------------------------------------------------
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
mkdir -p "$HF_CACHE"
HF_FREE=$(free_gb "$HF_CACHE")
OUT_FREE=$(free_gb "$SCRIPT_DIR")
echo "== Free space: HF cache disk ${HF_FREE} GB, output disk ${OUT_FREE} GB"
if [[ "$STREAM" != "1" && -n "$HF_FREE" && "$HF_FREE" -lt 80 ]]; then
    echo "!! Warning: less than 80 GB free for the HF cache. Set HF_HOME to a bigger disk, or use STREAM=1."
fi
if [[ -n "$OUT_FREE" && "$OUT_FREE" -lt 25 ]]; then
    echo "!! Warning: less than 25 GB free for the tokenized shards."
fi

# --- 3. FineWeb-Edu -------------------------------------------------------------
n_shards=$(ls "$SHARD_DIR"/edufineweb_*.npy 2>/dev/null | wc -l || true)
if [[ "$n_shards" -ge "$EXPECTED_SHARDS" ]]; then
    echo "== FineWeb-Edu already prepared ($n_shards shards in $SHARD_DIR), skipping"
else
    if [[ "$n_shards" -gt 0 ]]; then
        echo "== Found an incomplete $SHARD_DIR ($n_shards shards); removing it and starting over"
        rm -rf "$SHARD_DIR"
    fi
    echo "== Downloading + tokenizing FineWeb-Edu sample-10BT (this takes a few hours)"
    $PYTHON fineweb.py
    n_shards=$(ls "$SHARD_DIR"/edufineweb_*.npy | wc -l)
    echo "== Wrote $n_shards shards to $SHARD_DIR"
fi

# --- 4. HellaSwag ---------------------------------------------------------------
# (downloaded directly: importing hellaswag.py would require torch + transformers)
HS_FILE="hellaswag/hellaswag_val.jsonl"
HS_URL="https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
if [[ -s "$HS_FILE" ]]; then
    echo "== HellaSwag val set already present, skipping"
else
    echo "== Downloading HellaSwag val set"
    mkdir -p hellaswag
    $PYTHON -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$HS_URL" "$HS_FILE.tmp"
    mv "$HS_FILE.tmp" "$HS_FILE"
fi

# --- 5. optional cleanup --------------------------------------------------------
if [[ "${CLEAN_HF_CACHE:-0}" == "1" ]]; then
    echo "== Removing raw FineWeb-Edu files from the HF cache"
    rm -rf "$HF_CACHE"/hub/datasets--HuggingFaceFW--fineweb-edu \
           "$HF_CACHE"/datasets/HuggingFaceFW___fineweb-edu
fi

# --- 6. summary -------------------------------------------------------------------
echo
echo "== Done"
echo "   val shards:   $(ls "$SHARD_DIR"/edufineweb_val_*.npy | wc -l)"
echo "   train shards: $(ls "$SHARD_DIR"/edufineweb_train_*.npy | wc -l)"
echo "   shard size:   $(du -sh "$SHARD_DIR" | cut -f1)"
echo "   hellaswag:    $(ls -lh hellaswag/hellaswag_val.jsonl | awk '{print $5}')"
