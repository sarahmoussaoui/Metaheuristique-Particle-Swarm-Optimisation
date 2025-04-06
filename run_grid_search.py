import os
import csv
from src.modules.gridSearch import PSOGridSearch  # Your class file

# Folders to process
folders = ["data_20j_15m", "data_30j_15m", "data_50j_15m", "data_100j_20m"]
MAX_ITER = 100
root_dataset_dir = "src\data\processed"
csv_filename_all = "grid_search_results.csv"
csv_filename_best = "best_params_per_folder.csv"

csv_rows_all = []  # All param results
csv_rows_best = []  # Best param per folder
overall_makespan_per_combo = {}

# Define separate parameter grids for each folder
folder_param_grids = {
    "data_20j_15m": {
        "num_particles": [30, 50, 70],
        "w": [0.4, 0.7, 0.9],  # inertia weight
        "c1": [1.0, 1.5, 2.0],  # cognitive coefficient
        "c2": [1.0, 1.5, 2.0],  # social coefficient
    },
    "data_30j_15m": {
        "num_particles": [40, 70, 100],
        "w": [0.4, 0.7, 0.9],
        "c1": [1.5],
        "c2": [1.5, 2.0],
    },
    "data_50j_15m": {
        "num_particles": [100, 150],
        "w": [0.7, 0.9],
        "c1": [1.5],
        "c2": [1.5, 2.0],
    },
    "data_100j_20m": {
        "num_particles": [100],
        "w": [0.7, 0.9],
        "c1": [1.5],
        "c2": [2.0],
    },
}

for folder in folders:
    print(f"\n--- Running Grid Search in folder: {folder} ---")
    folder_path = os.path.join(root_dataset_dir, folder)
    param_grid = folder_param_grids[folder]

    # Init grid search
    search = PSOGridSearch(
        max_iter=MAX_ITER,
        dataset_folder=folder_path,
        output_file=f"{folder}_results.json",
    )
    search.set_parameter_grid(param_grid)

    # Run search
    result = search.run_search()

    for entry in result["search_history"]:
        params = entry["parameters"]
        avg_makespan = entry["avg_makespan"]

        combo_key = tuple(sorted(params.items()))
        if combo_key not in overall_makespan_per_combo:
            overall_makespan_per_combo[combo_key] = []
        overall_makespan_per_combo[combo_key].append(avg_makespan)

        csv_rows_all.append(
            {
                "folder": folder,
                **params,
                "avg_makespan": avg_makespan,
            }
        )

    # Save best result per folder
    best_params = result["best_parameters"]
    best_makespan = result["best_avg_makespan"]

    csv_rows_best.append(
        {
            "folder": folder,
            **best_params,
            "best_avg_makespan": best_makespan,
        }
    )

    print(
        f"Best parameters for {folder}: {best_params} => Avg makespan: {best_makespan:.2f}"
    )

# Write full parameter evaluation results
fieldnames_all = [
    "folder",
    "num_particles",
    "w",
    "c1",
    "c2",
    "avg_makespan",
]
with open(csv_filename_all, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames_all)
    writer.writeheader()
    writer.writerows(csv_rows_all)

# Write best parameters per folder
fieldnames_best = [
    "folder",
    "num_particles",
    "w",
    "c1",
    "c2",
    "best_avg_makespan",
]
with open(csv_filename_best, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames_best)
    writer.writeheader()
    writer.writerows(csv_rows_best)

# Optional: global average per param combo across all folders
print("\n--- Global Average Makespan Per Parameter Combination ---")
for combo_key, makespans in overall_makespan_per_combo.items():
    avg_across = sum(makespans) / len(makespans)
    print(f"{dict(combo_key)} => Avg Makespan Across Folders: {avg_across:.2f}")
