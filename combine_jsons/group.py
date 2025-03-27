import pandas as pd

# Load the CSV file
file_path = "results_dev_2025-03-11_17-06-07.csv"
results_df = pd.read_csv(file_path)

# Group by the specified columns and compute the mean
results_df = results_df.groupby(
    ["team_name", "submission_id", "task", "metric"]
).value.mean()

# Rename columns and reset index
results_df = results_df.add_suffix('_mean').reset_index()

# Save the processed DataFrame to a new CSV file
output_file = "processed_results.csv"
results_df.to_csv(output_file, index=False)

print(f"Processed file saved as {output_file}")
