"""Run one offline ChatTS Transformers inference and record a reproduction log."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "ckpt",
        help="Local ChatTS checkpoint directory (default: ./ckpt).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "reproduction_logs",
        help="Directory for timestamped logs.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens (default: 128).",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"transformers_smoke_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("chatts.smoke")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger, log_path


def move_inputs_to_device(inputs, device: torch.device):
    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(device)
    return inputs


def run(args: argparse.Namespace) -> int:
    logger, log_path = configure_logging(args.log_dir)
    model_path = args.model_path.expanduser().resolve()

    logger.info("Log file: %s", log_path)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Model path: %s", model_path)
    logger.info("Python: %s", sys.version.split()[0])
    logger.info("PyTorch: %s", torch.__version__)
    logger.info("Transformers: %s", transformers.__version__)
    logger.info("NumPy: %s", np.__version__)

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. ChatTS smoke test requires a GPU.")

    device = torch.device("cuda:0")
    logger.info("GPU: %s", torch.cuda.get_device_name(device))
    logger.info(
        "VRAM total: %.2f GB",
        torch.cuda.get_device_properties(device).total_memory / 1024**3,
    )

    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    logger.info("Loading model shards...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map=0,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    load_seconds = time.perf_counter() - load_started
    logger.info("Model loaded in %.2f seconds", load_seconds)
    logger.info(
        "VRAM after load: %.2f GB",
        torch.cuda.max_memory_allocated(device) / 1024**3,
    )

    logger.info("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        tokenizer=tokenizer,
    )

    length = 256
    x = np.arange(length)
    ts1 = np.sin(x / 10) * 5.0
    ts1[100:] -= 10.0
    ts2 = x * 0.05
    ts2[103] += 10.0

    question = (
        "I have two time series: TS1 <ts><ts/> and TS2 <ts><ts/>. "
        "Analyze their local changes and determine whether the changes "
        "occur near the same time. State the approximate points as evidence."
    )
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = processor(
        text=[prompt],
        timeseries=[ts1, ts2],
        padding=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs, device)

    logger.info("Generating with max_new_tokens=%d...", args.max_new_tokens)
    torch.cuda.reset_peak_memory_stats(device)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_started

    input_length = inputs["input_ids"].shape[1]
    answer = tokenizer.decode(
        outputs[0, input_length:],
        skip_special_tokens=True,
    ).strip()
    if not answer:
        raise RuntimeError("ChatTS returned an empty answer.")

    logger.info("Generation completed in %.2f seconds", generation_seconds)
    logger.info(
        "Generation peak VRAM: %.2f GB",
        torch.cuda.max_memory_allocated(device) / 1024**3,
    )
    logger.info("Question: %s", question)
    logger.info("Answer:\n%s", answer)
    logger.info("Smoke test PASSED")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception:
        logging.getLogger("chatts.smoke").exception("Smoke test FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
