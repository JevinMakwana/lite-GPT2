"""
Download a HellaSwag split and save it as hellaswag/hellaswag_<split>.jsonl,
the format hellaswag.py and train_gpt2.py read.

The original source (github.com/rowanz/hellaswag) is no longer available
(HTTP 451), so this pulls the same data from the HuggingFace mirror
Rowan/hellaswag (parquet) and converts it back to jsonl. The mirror stores
"label" as a string ("3"); it is converted to int here, because the training
loop compares it with an int prediction.

Needs only pyarrow (installed with `datasets`); no torch.
Run:  python download_hellaswag.py [val|train|test]
"""

import os
import sys
import json
import tempfile
import urllib.request

import pyarrow.parquet as pq

DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hellaswag")
HF_BASE = "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data"
HF_FILES = {
    "train": "train-00000-of-00001.parquet",
    "val": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
EXPECTED_ROWS = {"train": 39905, "val": 10042, "test": 10003}

def download(split="val"):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    out_path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    url = f"{HF_BASE}/{HF_FILES[split]}"
    print(f"Downloading {url}")
    with tempfile.TemporaryDirectory() as tmp:
        pq_path = os.path.join(tmp, "data.parquet")
        urllib.request.urlretrieve(url, pq_path)
        rows = pq.read_table(pq_path).to_pylist()
    expected = EXPECTED_ROWS[split]
    assert len(rows) == expected, f"expected {expected} rows, got {len(rows)}"
    tmp_out = out_path + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as f:
        for row in rows:
            # test split has no labels (empty string); keep it as-is there
            if row["label"] != "":
                row["label"] = int(row["label"])
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_out, out_path)
    print(f"Wrote {len(rows)} examples to {out_path}")
    return out_path

if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else "val")
