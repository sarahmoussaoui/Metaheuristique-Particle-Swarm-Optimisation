from src.modules.gridSearch import PSOGridSearch

if __name__ == "__main__":
    dataset_folder = "src/data/processed"
    output_file = "pso_grid_search_results.json"

    # Initialize grid search
    grid_search = PSOGridSearch(dataset_folder, output_file)

    # Run the grid search
    results = grid_search.run_search()

    print("\n=== Grid Search Completed ===")
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Average makespan: {results['best_avg_makespan']:.2f}")
    print(f"Results saved to {output_file}")
