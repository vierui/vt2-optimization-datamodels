# %%
import pandas as pd

# Specify input and output file paths
input_file = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed/load-winter.csv"  # Replace with your input file path
output_file = "/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed/load-winter.csv"  # Replace with your desired output file path

# Load the input file with the correct separator
df = pd.read_csv(input_file, sep=";")

# Convert the 'time' column to a proper datetime format
df['time'] = pd.to_datetime(df['time'], format="%d.%m.%y %H:%M")

# Rename columns
df.rename(columns={"load": "generation"}, inplace=True)

# Convert 'generation' column from scientific notation to float with desired scale
# df['generation'] = df['generation'] * 1e2

# Save the transformed file with the new format
df.to_csv(output_file, sep=";", index=False)

print(f"Transformed file saved as {output_file}")
# %%
