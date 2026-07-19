"""Convert one TelecomTS JSONL window to the ChatTS demo CSV layout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "demo" / "telecomts_smoke.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "demo" / "telecomts_example.csv"
DEFAULT_KPIS = (
    "RSRP",
    "UL_SNR",
    "DL_BLER",
    "UL_BLER",
    "DL_MCS",
    "UL_MCS",
    "TX_Bytes",
    "RX_Bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--line-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--kpis",
        nargs="+",
        default=list(DEFAULT_KPIS),
        help="KPI columns to export, in order.",
    )
    args = parser.parse_args()
    if args.line_index < 0:
        parser.error("--line-index must be non-negative")
    return args


def read_record(path: Path, line_index: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"TelecomTS JSONL file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == line_index:
                return json.loads(line)
    raise IndexError(f"No record at line index {line_index} in {path}")


def numeric_kpis(
    sample: dict[str, Any], requested_kpis: list[str]
) -> dict[str, list[int | float]]:
    kpis = sample.get("KPIs")
    if not isinstance(kpis, dict):
        raise ValueError("TelecomTS sample has no KPIs object")

    missing = [name for name in requested_kpis if name not in kpis]
    if missing:
        raise ValueError(f"Requested KPI channels are missing: {', '.join(missing)}")

    selected = {}
    for name in requested_kpis:
        values = kpis[name]
        if not (
            isinstance(values, list)
            and values
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            )
        ):
            raise ValueError(f"Requested KPI channel is not numeric: {name}")
        selected[name] = values
    if not selected:
        raise ValueError("TelecomTS sample contains no numeric KPI channels")
    lengths = {len(values) for values in selected.values()}
    if len(lengths) != 1:
        raise ValueError(f"Numeric KPI channels have inconsistent lengths: {lengths}")
    return selected


def write_csv(kpis: dict[str, list[int | float]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(kpis.keys())
        writer.writerows(zip(*kpis.values()))


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    sample = read_record(input_path, args.line_index)
    kpis = numeric_kpis(sample, args.kpis)
    write_csv(kpis, output_path)
    sequence_length = len(next(iter(kpis.values())))
    print(
        f"Wrote {len(kpis)} KPI columns x {sequence_length} rows to {output_path}"
    )


if __name__ == "__main__":
    main()
