import os
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
from itertools import product
from statistics import mean
from src.modules.jsspProcessor import JSSPProcessor


class PSOGridSearch:
    def __init__(
        self, max_iter, dataset_folder: str, output_file: str = "best_params.json"
    ):
        """
        Initialize the grid search with dataset folder and output file.

        Args:
            dataset_folder: Path to folder containing JSSP datasets
            output_file: File to save best parameters
        """
        self.dataset_folder = dataset_folder
        self.output_file = output_file
        self.best_results: Dict[str, Any] = {}
        self.max_iter = max_iter

        # Default parameter grid
        self.param_grid = {
            "num_particles": [30, 50, 100],
            "w": [0.4, 0.7, 0.9],  # inertia weight
            "c1": [1.0, 1.5, 2.0],  # cognitive coefficient
            "c2": [1.0, 1.5, 2.0],  # social coefficient
        }

        # Get all dataset files
        self.dataset_files = [
            os.path.join(dataset_folder, f)
            for f in os.listdir(dataset_folder)
            if f.endswith(".txt")
        ]

    def set_parameter_grid(self, grid: Dict[str, List[Any]]) -> None:
        """Set custom parameter grid for the search."""
        self.param_grid = grid

    def evaluate_parameter_set(
        self, params: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Evaluate a parameter set across all instances.

        Args:
            params: PSO parameters to evaluate

        Returns:
            Tuple of (average_makespan, average_execution_time, individual_results)
        """
        makespans = []
        exec_times = []
        individual_results = {}

        for dataset_path in self.dataset_files:
            filename = os.path.basename(dataset_path)
            processor = JSSPProcessor(dataset_path=dataset_path, plot=False)

            # Explicitly pass parameters from the dictionary
            _, makespan, exec_time = processor.run(
                num_particles=params["num_particles"],
                max_iter=self.max_iter,
                w=params["w"],
                c1=params["c1"],
                c2=params["c2"],
            )
            makespans.append(makespan)
            exec_times.append(exec_time)
            individual_results[filename] = {
                "makespan": makespan,
                "exec_time": exec_time,
            }

        return mean(makespans), mean(exec_times), individual_results

    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations from the grid."""
        param_names = self.param_grid.keys()
        value_combinations = product(*self.param_grid.values())
        return [dict(zip(param_names, combo)) for combo in value_combinations]

    def run_search(self) -> Dict[str, Any]:
        """Execute the grid search across all parameter combinations."""
        best_avg_makespan = float("inf")
        best_params = None
        best_avg_exec_time = float("inf")
        best_individual_results = {}
        search_history = []

        param_combinations = self.generate_parameter_combinations()
        total_combinations = len(param_combinations)

        print(
            f"Starting grid search with {total_combinations} parameter combinations "
            f"across {len(self.dataset_files)} instances"
        )

        for i, params in enumerate(param_combinations, 1):
            print(f"\nEvaluating combination {i}/{total_combinations}: {params}")

            avg_makespan, avg_exec_time, individual_results = (
                self.evaluate_parameter_set(params)
            )

            # Record this evaluation
            search_record = {
                "parameters": params,
                "avg_makespan": avg_makespan,
                "avg_exec_time": avg_exec_time,
                "individual_results": individual_results,
                "timestamp": datetime.now().isoformat(),
            }
            search_history.append(search_record)

            # Check if this is the best so far
            if avg_makespan < best_avg_makespan or (
                avg_makespan == best_avg_makespan and avg_exec_time < best_avg_exec_time
            ):
                best_avg_makespan = avg_makespan
                best_avg_exec_time = avg_exec_time
                best_params = params
                best_individual_results = individual_results

                print(f"New best parameters found! Avg makespan: {avg_makespan:.2f}")

                # Save intermediate results
                self.save_current_best(
                    best_params,
                    best_avg_makespan,
                    best_avg_exec_time,
                    best_individual_results,
                    search_history,
                )

        # Save final results
        self.best_results = {
            "best_parameters": best_params,
            "best_avg_makespan": best_avg_makespan,
            "best_avg_exec_time": best_avg_exec_time,
            "individual_results": best_individual_results,
            "search_history": search_history,
            "dataset_files": [os.path.basename(f) for f in self.dataset_files],
            "timestamp": datetime.now().isoformat(),
        }

        self.save_results()
        return self.best_results

    def save_current_best(
        self,
        params: Dict[str, Any],
        avg_makespan: float,
        avg_exec_time: float,
        individual_results: Dict[str, Any],
        search_history: List[Dict[str, Any]],
    ) -> None:
        """Save current best results during the search."""
        self.best_results = {
            "current_best_parameters": params,
            "current_avg_makespan": avg_makespan,
            "current_avg_exec_time": avg_exec_time,
            "current_individual_results": individual_results,
            "search_progress": {
                "completed": len(search_history),
                "total": len(self.generate_parameter_combinations()),
            },
            "timestamp": datetime.now().isoformat(),
        }
        self.save_results()

    def save_results(self) -> None:
        """Save results to JSON file."""
        with open(self.output_file, "w") as f:
            json.dump(self.best_results, f, indent=2)
