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

# 人工构造一条长度为256的时间序列：
# 前面是周期波动，从第100点开始整体下降10。
timeseries = np.sin(np.arange(256, dtype=np.float32) / 10.0) * 5.0
timeseries[100:] -= 10.0

question = (
    "I have a time series of length 256: <ts><ts/>. "
    "Briefly describe its periodic pattern and its main local change."
)

prompt = (
    "<|im_start|>system\n"
    "You are a helpful time series analysis assistant."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    f"{question}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

print("[4/4] 编码时间序列并生成回答...")

inputs = processor(
    text=[prompt],
    timeseries=[timeseries],
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
        max_new_tokens=200,
        do_sample=False,
        use_cache=True,
    )

answer = tokenizer.decode(
    outputs[0, input_length:],
    skip_special_tokens=True,
)

print("\n========== ChatTS回答 ==========")
print(answer)
print("================================")
