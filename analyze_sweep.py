import csv
import sys
import math
import statistics

if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    print("Usage: python analyze_sweep.py <csv_file>")
    sys.exit(1)

data = []
try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
             # Convert numeric fields
            clean_row = {}
            for k, v in row.items():
                if k == "Name":
                    clean_row[k] = v
                else:
                    try:
                        clean_row[k] = float(v)
                    except ValueError:
                        clean_row[k] = v
            data.append(clean_row)
except FileNotFoundError:
    print(f"Error: File {csv_file} not found")
    sys.exit(1)

if not data:
    print("No data found.")
    sys.exit(1)

# Sort by Policy Reward descending
data.sort(key=lambda x: x['Policy Reward'], reverse=True)

target_col = "Policy Reward"
excluded_keys = {"Name", "random_seed", target_col}
# Identify numeric parameters from the first row, hoping it's representative
params = [k for k in data[0].keys() if k not in excluded_keys and isinstance(data[0][k], (int, float))]

top_n = 10
top_performers = data[:top_n]

print(f"Top {top_n} performers ({target_col}: {top_performers[-1][target_col]:.2f} - {top_performers[0][target_col]:.2f}):")

for p in params:
    values = [run[p] for run in top_performers]
    min_val = min(values)
    max_val = max(values)
    mean_val = statistics.mean(values)
    median_val = statistics.median(values)
    print(f"{p}: range=[{min_val:.5g}, {max_val:.5g}], median={median_val:.5g}")

print() # Newline separator

def calculate_correlation(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("Lengths must equal")
    if n < 2:
        return 0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    sum_sq_diff_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_diff_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = math.sqrt(sum_sq_diff_x * sum_sq_diff_y)
    
    if denominator == 0:
        return 0
    return numerator / denominator

correlations = []
y_values = [d[target_col] for d in data]

for p in params:
    try:
        x_values = [d[p] for d in data]
        corr = calculate_correlation(x_values, y_values)
        correlations.append((p, corr))
    except (ValueError, TypeError):
        continue

# Sort by absolute correlation strength
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"Correlations with {target_col} (across {len(data)} runs):")
for p, corr in correlations:
    print(f"{p}: {corr:.4f}")
