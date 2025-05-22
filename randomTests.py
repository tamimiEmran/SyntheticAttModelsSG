
import os
import sys
import numpy as np
import pickle
from collections import defaultdict
from dataclasses import dataclass
#add project root to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --- Imports ---
import src.data.sgcc_data_preparation as sgcc 
sgcc_labels_path = sgcc.sgcc_labels_path


df = sgcc.load_dataframe("original")

#%%
import scipy.stats as st
import math
n=3
df = 2*n - 2  # 4
sd_all = 0.001
mean_all = 0.906

data = [
 ("7", 0.845, 0.050),
 ("5", 0.857, 0.032),
 ("9", 0.856, 0.031),
 ("0", 0.886, 0.049),
 ("1", 0.876, 0.021),
 ("8", 0.89375, 0.0125),
 ("2", 0.889, 0.011),
 ("10", 0.9, 0.01875),
 ("3", 0.9, 0.012),
 ("6", 0.903125, 0.015625),
 ("11", 0.906, 0.011),
 ("12", 0.906, 0.009),
 ("13", 0.906, 0.008)
]

results=[]
for attack, mean, sd in data:
    diff = mean - mean_all
    se = math.sqrt(sd_all**2 / n + sd**2 / n)
    t = diff / se
    p = 2*(1 - st.t.cdf(abs(t), df))
    results.append((attack, diff, p, t, se))
results

data = []
for attack, diff, p, t, se in results:
    data.append((attack, diff, p))

one_tail = []
for label, delta, p2 in data:
    if delta < 0:
        p1 = p2/2
    else:
        p1 = 1 - p2/2
    one_tail.append((label, delta, p1))

one_tail 
for label, delta, p1 in one_tail:
    print(f"{label}: {p1:.3f}")

# %%
