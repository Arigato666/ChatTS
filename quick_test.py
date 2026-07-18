from pathlib import Path
import importlib.util

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

MODEL_PATH = Path("/root/autodl-tmp/ChatTS/ckpt")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"模型目录不存在：{MODEL_PATH}")

if not torch.cuda.is_available():
    raise RuntimeError("当前环境没有检测到CUDA GPU")

# 安装了FlashAttention就使用它，否则先使用PyTorch SDPA跑通。
attention_backend = (
    "flash_attention_2"
    if importlib.util.find_spec("flash_attn") is not None
    else "sdpa"
)

print("模型目录：", MODEL_PATH)
print("GPU：", torch.cuda.get_device_name(0))
print("Attention后端：", attention_backend)

print("\n[1/4] 加载Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)

print("[2/4] 加载Processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    tokenizer=tokenizer,
    trust_remote_code=True,
    local_files_only=True,
)

print("[3/4] 加载ChatTS-8B模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map=0,
    low_cpu_mem_usage=True,
    attn_implementation=attention_backend,
)
model.eval()

def ask_chatts(question, series_list):
    prompt = (
        "<|im_start|>system\n"
        "你是一名时间序列分析助手，请始终使用中文简洁回答。"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = processor(
        text=[prompt],
        timeseries=series_list,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to("cuda:0") if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
        )

    return tokenizer.decode(
        outputs[0, input_length:],
        skip_special_tokens=True,
    )


# ========== 测试1：中文单变量 ==========
t = np.arange(256, dtype=np.float32)

series = np.sin(t / 10) * 5
series[100:] -= 10

answer1 = ask_chatts(
    "这是一条时间序列：<ts><ts/>。"
    "请分析它的周期性，并指出主要突变发生在什么位置。",
    [series],
)

print("\n========== 中文单变量 ==========")
print(answer1)


# ========== 测试2：中文多变量 ==========
cpu = 40 + 3 * np.sin(t / 10)
memory = 60 + 2 * np.sin(t / 10)

# 在第120点后同时上升
cpu[120:] += 20
memory[120:] += 15

answer2 = ask_chatts(
    "CPU使用率为：<ts><ts/>。"
    "内存使用率为：<ts><ts/>。"
    "请分析两条序列的主要变化，并判断它们是否存在同步关系。",
    [cpu, memory],
)

print("\n========== 中文多变量 ==========")
print(answer2)