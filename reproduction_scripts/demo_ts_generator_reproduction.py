#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2025 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[2]:


import sys
sys.path.append("..")  # Add parent directory to path
from chatts.ts_generator.generate import generate_random_attributes, generate_time_series, attribute_to_caption
import matplotlib.pyplot as plt
from pprint import pprint
import random
import numpy as np
from pathlib import Path

# 固定随机种子，保证每次生成相同的实验结果
random.seed(42)
np.random.seed(42)

# 创建结果保存目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reproduction_logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# In[72]:


# Generate a random attribute
SEQ_LEN = 256
attribute_pool = generate_random_attributes(seq_len=SEQ_LEN)
# Generate a time series with the random attribute
timeseries, attribute_pool = generate_time_series(attribute_pool, seq_len=SEQ_LEN)

# Plot time series
print(attribute_to_caption(timeseries, attribute_pool))
plt.figure(figsize=(10, 3))
plt.plot(timeseries)
print("\n========== Complete Attribute Pool ==========")
pprint(attribute_pool, sort_dicts=False)

plt.title("ChatTS Attribute-Based Synthetic Time Series")
plt.xlabel("Time point")
plt.ylabel("Value")
plt.grid(alpha=0.25)
plt.tight_layout()

figure_path = OUTPUT_DIR / "demo_ts_generator_sample.png"
plt.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nFigure saved to: {figure_path}")
print("Generator demo PASSED")