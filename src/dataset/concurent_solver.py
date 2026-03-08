import numpy as np
import cvxpy as cp
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing
import logging
import os



# Load data files
K =150
h = 1
p = 50
M = 30000

sale = pd.read_csv("")
lead_time = pd.read_csv("")
stock = pd.read_csv("")

param_dir = f"dataset/K={K}_h={h}_p={p}"
os.makedirs(param_dir, exist_ok=True)


file_paths = {
    "df_whether_ordering": f"{param_dir}/whether_order.csv",
    "df_stocking_level": f"{param_dir}/stock_level.csv",
    "df_back_order": f"{param_dir}/back_order.csv",
    "df_actrual_arrival": f"{param_dir}/actual_arrival.csv",
    "df_optimal_value": f"{param_dir}/optimal_value.csv",
    "df_optimal_order": f"{param_dir}/optimal_order.csv"
}
sku_col = sale.columns[0]  # usually 'sku_id'
other_cols = sale.columns[1:]

def make_empty_like_sale():
    df = pd.DataFrame({sku_col: sale[sku_col]})
    for col in other_cols:
        df[col] = np.nan
    return df

# Predefine properly initialized DataFrames
df_whether_ordering = make_empty_like_sale()
df_stocking_level   = make_empty_like_sale()
df_back_order       = make_empty_like_sale()
df_actrual_arrival  = make_empty_like_sale()
df_optimal_order    = make_empty_like_sale()

# Special case for df_optimal_value
df_optimal_value = pd.DataFrame({
    sku_col: sale[sku_col],
    'optimal_value': [np.nan] * len(sale)
})

# Mapping of DataFrame names to their initial values
initial_frames = {
    "df_whether_ordering": df_whether_ordering,
    "df_stocking_level": df_stocking_level,
    "df_back_order": df_back_order,
    "df_actrual_arrival": df_actrual_arrival,
    "df_optimal_value": df_optimal_value,
    "df_optimal_order": df_optimal_order
}

# Conditional loading or creation
dataframes = {}
for name, path in file_paths.items():
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
    else:
        df = initial_frames[name]
        df.to_csv(path, index=False)
    dataframes[name] = df

# Assign loaded DataFrames
df_whether_ordering = dataframes["df_whether_ordering"]
df_stocking_level = dataframes["df_stocking_level"]
df_back_order = dataframes["df_back_order"]
df_actrual_arrival = dataframes["df_actrual_arrival"]
df_optimal_value = dataframes["df_optimal_value"]
df_optimal_order = dataframes["df_optimal_order"]



T = len(sale.iloc[0].values[1:])


def find_index(arr, given_index, max_lead):
    result = []
    for i in range(max(0, given_index - max_lead), given_index):
        if i + arr[i] == given_index:
            result.append(i)
    return np.array(result)


def solve_optimization(i):
    if pd.isna(df_optimal_value.iat[i, 1]):
        q = cp.Variable(T, integer=True)
        y = cp.Variable(T, integer=True)
        a = cp.Variable(T, boolean=True)
        z = cp.Variable(T, integer=True)
        s = cp.Variable(T)

        D = sale.iloc[i].values[1:]  # Demand
        L = lead_time.iloc[i].values[1:]  # Lead times
        
        try:
            inventory_at_start = sale[sale['sku_id'] == lead_time.iloc[i].values[0]][lead_time.columns[1]].values[0]
            if inventory_at_start < 0:
                inventory_at_start = 0
        except:
            inventory_at_start = 0

        max_day_arrive = int(np.max(L))
        constraints = []
        objective = cp.Minimize(cp.sum(K * a + h * y + p * z))

        for t in range(1, T):
            order_arrive = find_index(L, t, max_day_arrive)
            constraints.append(s[t] <= D[t])
            constraints.append(s[t] <= y[t - 1] + cp.sum([q[t] for t in order_arrive]))
            constraints.append(y[t] == y[t - 1] - s[t] + cp.sum([q[t] for t in order_arrive]))
            constraints.append(z[t] == D[t] - s[t])

        for t in range(T):
            constraints.append(q[t] <= M * a[t])

        constraints.append(q >= 0)
        constraints.append(y >= 0)
        constraints.append(z >= 0)
        constraints.append(s >= 0)
        constraints.append(y[0] == inventory_at_start)

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.GUROBI, verbose=False, TimeLimit=180)
            if problem.status == 'optimal':
                actrual_arrival = np.zeros(T)
                for index, item in enumerate(a.value):
                    if np.allclose(1, item) and index + L[index] < len(a.value):
                        actrual_arrival[int(index + L[index])] = 1

                # Return the row index and computed results to the main process
                return (i, a.value, y.value, z.value, actrual_arrival, problem.value, q.value)
            else:
                print(f"Index {i}: Solver did not reach optimal solution. Status = {problem.status}")
                return None
        except Exception as e:
            return None
    return None


# Parallel execution
def run_parallel_optimizations():
    
    total_batches = len(sale) // 100
    with tqdm(total=total_batches) as pbar:
        # Run the parallel tasks in batches of 100
        
        with ProcessPoolExecutor(max_workers=5) as executor:
            
            for i in range(0, len(sale), 100):
                futures = [executor.submit(solve_optimization, j) for j in range(i, min(i + 100, len(sale)))]
                
                batch_results = [future.result() for future in futures if future.result()]
                
                # Update the dataframes with batch results
                for res in batch_results:
                    index, a_val, y_val, z_val, actual_arrival, opt_value, q_val = res
                    df_whether_ordering.iloc[index, 1:] = a_val
                    df_stocking_level.iloc[index, 1:] = y_val
                    df_back_order.iloc[index, 1:] = z_val
                    df_actrual_arrival.iloc[index, 1:] = actual_arrival
                    df_optimal_value.iloc[index, 1:] = np.array([opt_value])
                    df_optimal_order.iloc[index, 1:] = q_val

                # Save the updated dataframes after processing each batch of 100 rows
                df_whether_ordering.to_csv(f'{param_dir}/whether_order.csv', index=False)
                df_stocking_level.to_csv(f'{param_dir}/stock_level.csv', index=False)
                df_back_order.to_csv(f'{param_dir}/back_order.csv', index=False)
                df_actrual_arrival.to_csv(f'{param_dir}/actual_arrival.csv', index=False)
                df_optimal_value.to_csv(f'{param_dir}/optimal_value.csv', index=False)
                df_optimal_order.to_csv(f'{param_dir}/optimal_order.csv', index=False)
               
                # Update the progress bar
                pbar.update(1)

# Run the parallel optimizations
run_parallel_optimizations()
