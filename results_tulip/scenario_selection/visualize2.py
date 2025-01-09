import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib

folder = "WithMultipleRuns2"

# Create a figure and a set of subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Adjust figsize as needed

# List of files and their respective subplot positions
files = ["K100.1", "lin01", "P100.1", "wrp3-11"]
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # Positions in the 2x2 grid

methods = ["ffs"] + ["random"] * 25

# Iterate over files and their corresponding subplot positions
for file, pos in zip(files, positions):
    df = pd.DataFrame()
    for method_idx, method in enumerate(methods):
        # Construct the path to the CSV file
        path = f"project2/scenario_selection_graph/{folder}/{file}-50s_{method}_{method_idx+1}.csv"
        temp_df = pd.read_csv(path, sep=";")
        if len(df) == 0:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on="Sample Size", how="outer")

    if file == "K100.1" or file == "P100.1":
        last_row = {"Sample Size": 50}
        for column in df.columns:
            if "Objective" in column:
                last_row[column] = df["Objective ffs 1"].iloc[-1]
        df = df.append(last_row, ignore_index=True)

    df["Random avg"] = df[[f"Objective random {i}" for i in range(2, 27)]].mean(axis=1)
    df["Random min"] = df[[f"Objective random {i}" for i in range(2, 27)]].min(axis=1)
    df["Random max"] = df[[f"Objective random {i}" for i in range(2, 27)]].max(axis=1)
    # Plot in the respective subplot
    ax = axes[pos[0], pos[1]]  # Access the correct subplot

    ax.plot(df["Sample Size"], df["Objective ffs 1"], "-o", label="Fast Forward Selection")
    ax.plot(df["Sample Size"], df["Random avg"], "-o", label="Random Sample")
    # Plot the std as deviation
    ax.fill_between(df["Sample Size"], df["Random min"], df["Random max"], color='gray', alpha=0.2)

    ax.set_xlabel("Sample Size")
    ax.set_ylabel("Objective Value")
    ax.set_title(file)
    # ax.set_yscale("log")  # Set y-axis to log scale

# Adjust layout and save the entire figure
plt.tight_layout()

# plt.legend()
tikzplotlib.save("project2/scenario_selection_graph/combined_plot_final.tex")

plt.show()
