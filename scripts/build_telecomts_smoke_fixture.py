"""Build a two-window TelecomTS fixture for ChatTS smoke testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "telecomTS" / "data" / "raw" / "TelecomTS"
DEFAULT_OUTPUT = PROJECT_ROOT / "demo" / "telecomts_smoke.jsonl"

SOURCE_FILES = {
    "anomalous": Path("anomalous/jammer/File/processed/chunked.jsonl"),
    "normal": Path(
        "normal/stationary/Zone_A/no_congestion/File/processed/chunked.jsonl"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_first_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"TelecomTS source file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        line = handle.readline()
    if not line:
        raise RuntimeError(f"TelecomTS source file is empty: {path}")
    return json.loads(line)


def compact_qna(sample: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    compact: dict[str, list[dict[str, Any]]] = {}
    for category in ("timeseries", "network", "anomalies"):
        entries = sample.get("QnA", {}).get(category, [])
        if entries:
            compact[category] = [
                {"q": entries[0].get("q"), "a": entries[0].get("a")}
            ]
    return compact


def compact_record(
    sample: dict[str, Any], scenario: str, relative_source: Path
) -> dict[str, Any]:
    anomaly = sample.get("anomalies") or {}
    return {
        "fixture_metadata": {
            "source_dataset": "AliMaatouk/TelecomTS",
            "source_file": relative_source.as_posix(),
            "source_line": 0,
            "scenario": scenario,
            "license": "MIT",
            "purpose": "ChatTS smoke test only",
        },
        "start_time": sample.get("start_time"),
        "end_time": sample.get("end_time"),
        "sampling_rate": sample.get("sampling_rate"),
        "KPIs": sample.get("KPIs"),
        "labels": sample.get("labels"),
        "anomalies": {
            "exists": anomaly.get("exists"),
            "type": anomaly.get("type"),
            "anomaly_duration": anomaly.get("anomaly_duration"),
            "affected_kpis": anomaly.get("affected_kpis"),
        },
        "QnA": compact_qna(sample),
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    records = []
    for scenario, relative_source in SOURCE_FILES.items():
        sample = read_first_record(data_root / relative_source)
        records.append(compact_record(sample, scenario, relative_source))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    print(f"Wrote {len(records)} TelecomTS smoke-test records to {output}")


if __name__ == "__main__":
    main()
