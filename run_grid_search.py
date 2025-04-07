from src.modules.gridSearch import PSOGridSearch


def main():
    # Configuration
    max_iter = 2000  # Number of iterations for each PSO run
    dataset_folder = (
        "src/data/processed/data_20j_15m"  # Folder containing your JSSP instance files
    )
    output_params = "best_params_grid_1.json"  # Output file for best parameters
    output_history = "search_history_grid_1.csv"  # Output file for search history

    # Initialize and run grid search
    grid_search = PSOGridSearch(
        max_iter=max_iter,
        dataset_folder=dataset_folder,
        params_output_file=output_params,
        history_output_file=output_history,
        num_samples=10,  # Number of samples to take from parameter ranges
    )

    # Optional: Customize parameter grid if needed
    grid_search.set_parameter_grid(
        {
            "num_particles": [20, 30, 50],  # Specific values
            "w": (0.4, 0.9),  # Range
            "c1": [0.5, 1, 1.5, 2],  # Fixed value
            "c2": [0.5, 1, 1.5, 2],  # Fixed value
            "mutation_rate": [0.1, 0.2, 0.3],  # Fixed
            "max_stagnation": [20],
            "early_stopping_window": [None],  # Fixed
            "improvement_threshold": [0.005],
        },
    )

    print("Starting grid search...")
    results = grid_search.run_search()

    print("\nGrid search completed!")
    print(f"Best parameters saved to: {output_params}")
    print(f"Search history saved to: {output_history}")
    print(f"Best average makespan: {results['best_avg_makespan']:.2f}")


if __name__ == "__main__":
    main()
