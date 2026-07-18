# TelecomTS smoke test

`telecomts_smoke.jsonl` contains two derived TelecomTS windows: one real
jamming sample and one context-matched normal sample. Both use File traffic,
Zone A, stationary operation, and no congestion. The fixture keeps the raw KPI
values, evaluation labels, and one Q&A per category. Long descriptions,
reasoning traces, and troubleshooting tickets are intentionally excluded.

The source dataset is
[`AliMaatouk/TelecomTS`](https://huggingface.co/datasets/AliMaatouk/TelecomTS)
and is distributed under the MIT license. The fixture is only intended for
integration and inference smoke tests, not benchmark reporting.

`telecomts_example.csv` is the anomalous fixture window in the same layout as
`ts_example.csv`: one numeric KPI per column and one time point per row. It has
16 columns and 128 rows, with no index or timestamp column, so it can be
uploaded directly through the ChatTS CSV demo. The two categorical protocol
channels are excluded because the current ChatTS processor accepts numeric
time series.

`telecomts_synthetic_example.csv` is a second CSV from the synthetic TelecomTS
anomaly set. It is an `Antenna Failure` window with the annotated anomaly range
`46-98`, so it is more suitable for inspecting an onset and recovery pattern.
The Jamming CSV is still useful for whole-window anomaly recognition, but it
does not provide an onset boundary because its annotation covers `0-127`.

## Server commands

From the ChatTS repository root, validate the two records without loading the
model:

```bash
python -m demo.telecomts_hf \
  --source-file demo/telecomts_smoke.jsonl \
  --task anomaly \
  --max-samples 2 \
  --dry-run
```

Run ChatTS-8B inference:

```bash
python -m demo.telecomts_hf \
  --model-path /root/autodl-tmp/ChatTS/ckpt \
  --source-file demo/telecomts_smoke.jsonl \
  --task anomaly \
  --max-samples 2 \
  --output reproduction_outputs/telecomts_smoke_results.jsonl
```

Run a TelecomTS time-series Q&A instead:

```bash
python -m demo.telecomts_hf \
  --model-path /root/autodl-tmp/ChatTS/ckpt \
  --source-file demo/telecomts_smoke.jsonl \
  --task qna \
  --qna-category timeseries \
  --max-samples 2
```

Rebuild the fixture from a full local TelecomTS download:

```bash
python scripts/build_telecomts_smoke_fixture.py \
  --data-root /path/to/TelecomTS
```

Convert any fixture or raw `chunked.jsonl` record to the demo CSV layout:

```bash
python scripts/telecomts_jsonl_to_csv.py \
  --input demo/telecomts_smoke.jsonl \
  --line-index 0 \
  --output demo/telecomts_example.csv
```

For the synthetic onset example, use the raw TelecomTS file and its selected
record:

```bash
python scripts/telecomts_jsonl_to_csv.py \
  --input /path/to/TelecomTS/anomalous/synthetic/Zone_A/File/processed/chunked.jsonl \
  --line-index 19 \
  --output demo/telecomts_synthetic_example.csv
```
