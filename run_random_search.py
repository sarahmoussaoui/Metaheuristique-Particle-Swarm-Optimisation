import os
import csv
import json
from src.modules.randomSearch import PSORandomSearch

# Configuration
MAX_ITER = 100
NUM_SAMPLES = 50  # Number of parameter combinations to try
root_dataset_dir = "src/data/processed"
folders = ["data_20j_15m"]

# Recommended parameter specifications for search
folder_param_specs = {
    "data_20j_15m": {
        "num_particles": [20, 30, 50, 70, 100],  # Specific values
        "w": (0.4, 0.9),  # Range
        "c1": [0.5, 1, 1.5, 2],  # Fixed value
        "c2": [0.5, 1, 1.5, 2],  # Fixed value
        "mutation_rate": [0.1, 0.2, 0.3],  # Fixed
        "max_stagnation": (20, 60),  # Range
        "early_stopping_window": 40,  # Fixed
        "improvement_threshold": [0.005, 0.01],
    },
}

# Prepare output directory
os.makedirs("random_search_results", exist_ok=True)

# Create summary CSV
summary_path = "random_search_results/summary.csv"
with open(summary_path, "w", newline="") as summary_file:
    writer = csv.writer(summary_file)
    writer.writerow(
        [
            "Folder",
            "Best Makespan",
            "Num Particles",
            "Inertia (w)",
            "Cognitive (c1)",
            "Social (c2)",
            "Mutation Rate",
            "Max Stagnation",
            "Early Stop Window",
            "Improvement Threshold",
        ]
    )

for folder in folders:
    print(f"\n=== Running Parameter Search in {folder} ===")
    folder_path = os.path.join(root_dataset_dir, folder)

    search = PSORandomSearch(
        max_iter=MAX_ITER,
        dataset_folder=folder_path,
        output_prefix=f"random_search_results/{folder}",
        num_samples=NUM_SAMPLES,
        seed=42,
    )

    # Set mixed parameter specifications
    search.set_parameter_specs(folder_param_specs[folder])

    # Run the search
    best_results = search.run_search()
    best_params = best_results["best_parameters"]

    # Print and save summary
    print(f"\nBest parameters for {folder}:")
    print(json.dumps(best_params, indent=2))
    print(f"Average makespan: {best_results['best_avg_makespan']:.2f}")

    # Append to summary CSV
    with open(summary_path, "a", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            [
                folder,
                f"{best_results['best_avg_makespan']:.2f}",
                best_params["num_particles"],
                best_params["w"],
                best_params["c1"],
                best_params["c2"],
                best_params["mutation_rate"],
                best_params["max_stagnation"],
                best_params["early_stopping_window"],
                best_params["improvement_threshold"],
            ]
        )

print("\nSearch completed! Results saved in:")
print(f"- Individual JSON/CSV files in 'random_search_results' folder")
print(f"- Summary of best parameters in 'random_search_results/summary.csv'")
