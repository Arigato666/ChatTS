"""Run ChatTS inference on raw TelecomTS JSONL windows.

Examples:
    python -m demo.telecomts_hf --dry-run
    python -m demo.telecomts_hf --task anomaly --max-samples 2
    python -m demo.telecomts_hf --task qna --qna-category timeseries
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    os.environ.get("CHATTS_MODEL_PATH", "/root/autodl-tmp/ChatTS/ckpt")
)
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "telecomTS" / "data" / "raw" / "TelecomTS"

KPI_UNITS = {
    "RSRP": "dB",
    "DL_BLER": "ratio",
    "DL_MCS": "index",
    "UL_BLER": "ratio",
    "UL_MCS": "index",
    "UL_NPRB": "count",
    "UL_SNR": "dB",
    "TX_Bytes": "bytes",
    "RX_Bytes": "bytes",
    "Estimated_UL_Buffer": "KB",
    "PRBs_DL_Current": "count",
    "PRBs_UL_Current": "count",
    "PRB_Utilization_DL": "percent",
    "PRB_Utilization_UL": "percent",
    "UL_NumberOfPackets": "count",
    "DL_NumberOfPackets": "count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use TelecomTS windows as native time-series inputs to ChatTS."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--source-file",
        type=Path,
        help="One chunked.jsonl file. Relative paths are resolved under --data-root.",
    )
    parser.add_argument(
        "--scenario",
        choices=("all", "normal", "anomalous"),
        default="all",
        help="Filter files by their path when --source-file is not set.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument(
        "--task", choices=("describe", "anomaly", "qna"), default="anomaly"
    )
    parser.add_argument(
        "--qna-category",
        choices=("timeseries", "network", "anomalies"),
        default="timeseries",
    )
    parser.add_argument("--qna-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "telecomts_chatts_results.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print inputs without loading ChatTS.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to fetch missing model files.",
    )
    args = parser.parse_args()

    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if args.qna_index < 0:
        parser.error("--qna-index must be non-negative")
    return args


def resolve_source_files(
    data_root: Path, source_file: Path | None, scenario: str
) -> list[Path]:
    data_root = data_root.expanduser().resolve()
    if source_file is not None:
        source_file = source_file.expanduser()
        if source_file.is_absolute():
            files = [source_file.resolve()]
        else:
            candidates = [
                (Path.cwd() / source_file).resolve(),
                (PROJECT_ROOT / source_file).resolve(),
                (data_root / source_file).resolve(),
            ]
            files = [next((path for path in candidates if path.is_file()), candidates[0])]
    else:
        search_root = data_root if scenario == "all" else data_root / scenario
        files = sorted(search_root.rglob("chunked.jsonl"))

    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"TelecomTS JSONL file does not exist: {missing[0]}")
    if not files:
        raise FileNotFoundError(
            f"No chunked.jsonl files found for scenario={scenario!r} under {data_root}"
        )
    return files


def iter_samples(files: Iterable[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if line.strip():
                    yield path, line_index, json.loads(line)


def select_samples(
    files: Iterable[Path], sample_index: int, max_samples: int
) -> list[tuple[Path, int, dict[str, Any]]]:
    selected = []
    for global_index, item in enumerate(iter_samples(files)):
        if global_index < sample_index:
            continue
        selected.append(item)
        if len(selected) == max_samples:
            break
    if not selected:
        raise IndexError(f"No sample exists at global sample index {sample_index}")
    return selected


def extract_numeric_series(sample: dict[str, Any]) -> tuple[list[str], list[np.ndarray]]:
    kpis = sample.get("KPIs")
    if not isinstance(kpis, dict):
        raise ValueError("TelecomTS sample has no KPIs object")

    names: list[str] = []
    series: list[np.ndarray] = []
    expected_length: int | None = None
    skipped: list[str] = []
    for name, values in kpis.items():
        try:
            array = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError):
            skipped.append(name)
            continue
        if array.ndim != 1:
            raise ValueError(f"KPI {name!r} must be one-dimensional, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"KPI {name!r} contains NaN or infinite values")
        if expected_length is None:
            expected_length = len(array)
        elif len(array) != expected_length:
            raise ValueError(
                f"KPI {name!r} has length {len(array)}, expected {expected_length}"
            )
        names.append(name)
        series.append(array)

    if not series:
        raise ValueError("TelecomTS sample contains no numeric KPI series")
    if len(series) > 30:
        raise ValueError(f"ChatTS recommends at most 30 series, found {len(series)}")
    if skipped:
        print(f"Skipped non-numeric KPI channels: {', '.join(skipped)}")
    return names, series


def get_question_and_reference(
    sample: dict[str, Any], task: str, qna_category: str, qna_index: int
) -> tuple[str, Any]:
    if task == "describe":
        return (
            "Describe the important global and local behavior of these KPI series, "
            "then identify notable relationships across KPIs. Do not make a diagnosis "
            "that is not supported by the signals.",
            None,
        )
    if task == "anomaly":
        question = (
            "Based only on the numeric KPI series, determine whether this window is "
            "anomalous. These signals come from a live 5G observability system and can "
            "be naturally noisy and bursty. Start with exactly 'Anomaly: Yes' or "
            "'Anomaly: No', then briefly state the evidence across KPIs. Do not assume "
            "that every spike is abnormal."
        )
        reference = sample.get("labels", {}).get("anomaly_present")
        return question, reference

    entries = sample.get("QnA", {}).get(qna_category, [])
    if qna_index >= len(entries):
        raise IndexError(
            f"QnA category {qna_category!r} has {len(entries)} entries; "
            f"index {qna_index} is unavailable"
        )
    entry = entries[qna_index]
    return entry["q"], entry.get("a")


def build_question(
    names: list[str], series: list[np.ndarray], sampling_rate: Any, question: str
) -> str:
    length = len(series[0])
    try:
        rate_text = f"{float(sampling_rate):g} Hz"
    except (TypeError, ValueError):
        rate_text = "an unknown rate"
    lines = [
        f"I have {len(series)} numeric 5G KPI time series sampled at {rate_text}; "
        f"each series has length {length}."
    ]
    for index, name in enumerate(names, start=1):
        unit = KPI_UNITS.get(name, "native units")
        lines.append(f"Series {index} ({name}, {unit}): <ts><ts/>")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def load_chatts(model_path: Path, gpu: int, allow_download: bool) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ChatTS inference")
    local_files_only = not allow_download
    attention_backend = (
        "flash_attention_2"
        if importlib.util.find_spec("flash_attn") is not None
        else "sdpa"
    )
    print(f"Loading ChatTS from {model_path} on cuda:{gpu} ({attention_backend})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=torch.float16,
        device_map=gpu,
        low_cpu_mem_usage=True,
        attn_implementation=attention_backend,
    )
    model.eval()
    return model, tokenizer, processor


def format_chat_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful time series analysis assistant."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_answer(
    model: Any,
    tokenizer: Any,
    processor: Any,
    question: str,
    series: list[np.ndarray],
    gpu: int,
    max_new_tokens: int,
) -> str:
    import torch

    prompt = format_chat_prompt(tokenizer, question)
    inputs = processor(
        text=[prompt],
        timeseries=series,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(f"cuda:{gpu}") if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    return tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True).strip()


def build_record(
    source_file: Path,
    line_index: int,
    sample: dict[str, Any],
    names: list[str],
    question: str,
    reference: Any,
    answer: str | None,
) -> dict[str, Any]:
    return {
        "source_file": str(source_file),
        "line_index": line_index,
        "start_time": sample.get("start_time"),
        "end_time": sample.get("end_time"),
        "sampling_rate": sample.get("sampling_rate"),
        "kpi_names": names,
        "question": question,
        "reference_answer": reference,
        "model_answer": answer,
        "labels_for_evaluation_only": sample.get("labels"),
        "anomaly_type_for_evaluation_only": sample.get("anomalies", {}).get("type"),
    }


def main() -> None:
    args = parse_args()
    files = resolve_source_files(args.data_root, args.source_file, args.scenario)
    selected = select_samples(files, args.sample_index, args.max_samples)

    prepared = []
    for source_file, line_index, sample in selected:
        names, series = extract_numeric_series(sample)
        question, reference = get_question_and_reference(
            sample, args.task, args.qna_category, args.qna_index
        )
        model_question = build_question(
            names, series, sample.get("sampling_rate"), question
        )
        prepared.append(
            (source_file, line_index, sample, names, series, model_question, reference)
        )

    if args.dry_run:
        for source_file, line_index, sample, names, series, question, reference in prepared:
            print("\n=== TelecomTS dry run ===")
            print(f"Source: {source_file} (line {line_index})")
            print(f"Numeric inputs: {len(series)} x {len(series[0])}")
            print(f"Reference answer: {reference}")
            print(question)
        return

    model, tokenizer, processor = load_chatts(
        args.model_path.expanduser(), args.gpu, args.allow_download
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for item_index, (
            source_file,
            line_index,
            sample,
            names,
            series,
            question,
            reference,
        ) in enumerate(prepared, start=1):
            answer = generate_answer(
                model,
                tokenizer,
                processor,
                question,
                series,
                args.gpu,
                args.max_new_tokens,
            )
            record = build_record(
                source_file,
                line_index,
                sample,
                names,
                question,
                reference,
                answer,
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"\n=== Result {item_index}/{len(prepared)} ===")
            print(f"Reference: {reference}")
            print(f"ChatTS: {answer}")
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
