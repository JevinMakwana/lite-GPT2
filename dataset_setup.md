# Dataset Setup

## Run the data pipeline

From the project folder (`build-nanogpt-master/`) on the Linux server:

```bash
bash setup_data.sh
```

This downloads and prepares every dataset `train_gpt2.py` needs. It takes a few hours, so run it inside `tmux` or `screen` so it survives an SSH disconnect. It is safe to re-run: finished steps are skipped.

### Without the HuggingFace cache (recommended if disk is tight)

```bash
STREAM=1 bash setup_data.sh
```

By default, HuggingFace `load_dataset` first saves the raw parquet files to `~/.cache/huggingface`, then converts them into a second, arrow-format copy for fast random access. Together that is ~70–80 GB. This project only reads the data **once, in order**, to write the token shards, so neither copy is needed afterwards.

`STREAM=1` reads the parquet files straight from HuggingFace over the network and tokenizes them as they arrive. Only the ~20 GB of token shards in `edu_fineweb10B/` are written to disk.

Trade-off: if a streamed run is interrupted (network drop, SSH disconnect without `tmux`), it starts over from the beginning, because nothing was cached. Use `tmux`.

Optional settings:

| Variable | Effect |
|---|---|
| `STREAM=1` | Stream FineWeb-Edu instead of caching it (no ~70–80 GB cache) |
| `HF_HOME=/big/disk/hf_cache` | Where HuggingFace stores the raw download when not streaming (needs ~70–80 GB). Default: `~/.cache/huggingface` |
| `CLEAN_HF_CACHE=1` | Delete the raw FineWeb-Edu download after tokenizing |
| `SKIP_PIP=1` | Don't pip-install dependencies |

Example:

```bash
HF_HOME=/path/to/big/disk/hf_cache CLEAN_HF_CACHE=1 bash setup_data.sh
```

## Datasets used

### 1. FineWeb-Edu (`sample-10BT`): training and validation loss

- **Source:** [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), a 10-billion-token sample of web pages filtered for educational quality.
- **Purpose:** the pretraining corpus. The model learns to predict the next token on this text.
- **Prepared by:** `fineweb.py` (called by `setup_data.sh`), which tokenizes it with the GPT-2 tokenizer (`tiktoken`).
- **Output:** `edu_fineweb10B/`, 100 shards of 100M tokens each (~20 GB, stored as `uint16`).
  - `edufineweb_val_000000.npy`: the first shard, held out to measure **validation loss** every 250 steps.
  - `edufineweb_train_000001.npy` … `000099.npy`: the 99 **training** shards.

### 2. HellaSwag (validation split): benchmark evaluation

- **Source:** [rowanz/hellaswag](https://github.com/rowanz/hellaswag), 10,042 multiple-choice sentence-completion questions.
- **Purpose:** an evaluation only, never trained on. Every 250 steps the model picks the most likely of 4 endings per question, and the accuracy is logged to track how well the model understands common-sense text. Reference: OpenAI's GPT-2 124M scores ~29.5% with this script's method.
- **Prepared by:** `hellaswag.py` (`download('val')`, called by `setup_data.sh`).
- **Output:** `hellaswag/hellaswag_val.jsonl` (~11 MB).

## Disk space

| What | Size |
|---|---|
| HuggingFace cache (raw download + processed copy) | ~70–80 GB (estimate); **0 with `STREAM=1`**, or removable afterwards with `CLEAN_HF_CACHE=1` |
| `edu_fineweb10B/` tokenized shards | ~20 GB |
| `hellaswag/` | ~11 MB |
