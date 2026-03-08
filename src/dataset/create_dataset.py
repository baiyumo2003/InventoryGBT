import numpy as np
import json
import os
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

base_dir = ""
window_size=60

demand=pd.read_csv("")
leadtime=pd.read_csv("")
actual_arrival = pd.read_csv(os.path.join(base_dir, "actual_arrival.csv"))
back_order = pd.read_csv(os.path.join(base_dir, "back_order.csv"))
optimal_order = pd.read_csv(os.path.join(base_dir, "optimal_order.csv"))
stock_level = pd.read_csv(os.path.join(base_dir, "stock_level.csv"))
whether_order = pd.read_csv(os.path.join(base_dir, "whether_order.csv"))

def distance_to_next_one(binary_list):
    n = len(binary_list)
    result = [999] * n
    next_one_idx = -1 

    for i in range(n - 1, 0, -1):
        if binary_list[i] == 1:
            result[i] = 0
            next_one_idx = i
        elif next_one_idx != -1:
            result[i] = next_one_idx - i

    return result


def fill_zeros_from_right(arr):
    n = len(arr)
    result = arr.copy()
    next_val = 0
    for i in range(n - 1, -1, -1):
        if abs(result[i]) > 1e-8:  # Treat -0.0 or 0.0 as zero
            next_val = result[i]
        elif next_val is not None:
            result[i] = next_val
    return result

data = {
    "demand":demand,
    "leadtime":leadtime,
    "actual_arrival": actual_arrival.apply(lambda row: distance_to_next_one(row.tolist()), axis=1, result_type='expand'),
    "back_order": back_order,
    "optimal_order": optimal_order.apply(lambda row: fill_zeros_from_right(row.tolist()), axis=1, result_type='expand'),
    #  "optimal_order": optimal_order,
    "stock_level": stock_level,
    "whether_order": whether_order,
}
T, N = data["demand"].shape



rows = []
for i in tqdm(range(T)):
    avg_order=data["optimal_order"].iloc[i,1:].mean()
    if avg_order<15:
        print("Not good",i)
        continue
    for j in range(1, N - window_size):
        sample = ""  # Reset for each row i
        for k in range(j,j+window_size):
            d = int(round(float(data["demand"].iloc[i, k])))
            l = int(round(float(data["leadtime"].iloc[i, k])))
            b = int(round(float(data["back_order"].iloc[i, k])))
            s = int(round(float(data["stock_level"].iloc[i, k])))
            obs = f"[OBS] {d} {l} {b} {s} "
            
            next_arrival=int(round(float(data["actual_arrival"].iloc[i, k])))
            arrival_quantity=int(round(float(data["optimal_order"].iloc[i, k])))
            act=f"[ACT] {l} {next_arrival} {arrival_quantity} "
            
            sample += obs  # Append current obs to the sample
            sample += act
        rows.append({"text": sample})  # Add completed sample for this i


df=pd.DataFrame(rows)
hf_dataset = Dataset.from_pandas(df)

# Save the dataset to disk for later use
hf_dataset.save_to_disk(os.path.join(base_dir,"inventory_gpt2_dataset_hf"))
    
    