from src.modules.gridSearch import PSOGridSearch


def main():
    # Configuration
    dataset_file = "src/data/processed/data_50j_15m/data_100j_20m_1.txt"  # Folder containing your JSSP instance files
    output_params = "SPSO/paramaters/best_params_grid_100j_20m_1.json"  # Output file for best parameters
    output_history = "SPSO/paramaters/search_history_grid_100j_20m_1.csv"  # Output file for search history

    # Initialize and run grid search
    grid_search = PSOGridSearch(
        dataset_file=dataset_file,
        params_output_file=output_params,
        history_output_file=output_history,
    )

    # Optional: Customize parameter grid if needed
    grid_search.set_parameter_grid(
        {
            "num_particles": [100, 150, 200],
            "max_iter": [1000],
            "w": [(0.2, 0.8), (0.4, 0.9)],  # (min, max) tuples
            "c1": [(0.2, 0.9), (0.4, 0.9)],  # (min, max) tuples
            "c2": [(0.2, 0.8), (0.3, 0.7)],  # (min, max) tuples
            "mutation_rate": [0.5, 0.7],
            "max_stagnation": [20],
            "early_stopping_window": [None],
            "improvement_threshold": [0.05],
        }
    )
    # Optional: Customize parameter grid if needed

    print("Starting grid search...")
    results = grid_search.run_search()

    print("\nGrid search completed!")
    print(f"Best parameters saved to: {output_params}")
    print(f"Search history saved to: {output_history}")


if __name__ == "__main__":
    main()
