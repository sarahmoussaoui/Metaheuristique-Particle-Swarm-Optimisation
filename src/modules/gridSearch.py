import os
import json
import csv
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union
from itertools import product
from statistics import mean
from src.modules.jsspProcessor import JSSPProcessor


class PSOGridSearch:
    def __init__(
        self,
        max_iter: int,
        dataset_folder: str,
        params_output_file: str = "best_params.json",
        history_output_file: str = "search_history_grid.csv",
        num_samples: int = 50,  # Number of samples to take from continuous ranges
    ):
        """
        Initialize the grid search with dataset folder and output files.

        Args:
            max_iter: Maximum iterations for PSO
            dataset_folder: Path to folder containing JSSP datasets
            params_output_file: JSON file to save best parameters
            history_output_file: CSV file to save search history
            num_samples: Number of samples to take from continuous parameter ranges
        """
        self.dataset_folder = dataset_folder
        self.params_output_file = params_output_file
        self.history_output_file = history_output_file
        self.max_iter = max_iter
        self.num_samples = num_samples
        self.best_results: Dict[str, Any] = {}
        self.search_history: List[Dict[str, Any]] = []

        # Updated parameter grid with ranges and exact values
        self.param_grid = {
            "num_particles": (20, 50),  # Range
            "w": [0.4, 0.6, 0.8],  # List of exact values
            "c1": (1.0, 2.0),  # Range
            "c2": (1.0, 2.0),  # Range
            "mutation_rate": 0.1,  # Exact value
            "max_stagnation": (10, 30),  # Range
            "early_stopping_window": 70,  # List of exact values
            "improvement_threshold": 0.01,  # Exact value
        }

        # Validate and get dataset files
        self._validate_dataset_folder()
        self.dataset_files = self._get_dataset_files()

    def _validate_dataset_folder(self) -> None:
        """Ensure dataset folder exists and is accessible."""
        if not os.path.isdir(self.dataset_folder):
            raise ValueError(f"Dataset folder not found: {self.dataset_folder}")

    def _get_dataset_files(self) -> List[str]:
        """Get all dataset files from the folder."""
        return [
            os.path.join(self.dataset_folder, f)
            for f in os.listdir(self.dataset_folder)
            if f.endswith(".txt")
        ]

    def set_parameter_grid(
        self, grid: Dict[str, Union[List[Any], Tuple[Any, Any], Any]]
    ) -> None:
        """Set custom parameter grid for the search."""
        self.param_grid = grid

    def _sample_parameter_space(
        self, param_name: str, param_values: Union[List[Any], Tuple[Any, Any], Any]
    ) -> List[Any]:
        """
        Generate samples for a parameter based on its specification.

        Args:
            param_name: Name of the parameter (for error messages)
            param_values: Either a list, tuple (range), or single value

        Returns:
            List of sampled values for this parameter
        """
        if isinstance(param_values, list):
            # Exact values provided as list
            return param_values
        elif isinstance(param_values, tuple):
            # Range provided as tuple (min, max)
            if len(param_values) != 2:
                raise ValueError(
                    f"Range for {param_name} must be a tuple of (min, max)"
                )
            min_val, max_val = param_values
            if min_val >= max_val:
                raise ValueError(
                    f"Invalid range for {param_name}: min must be less than max"
                )

            # Generate samples from the range
            if param_name in [
                "num_particles",
                "max_stagnation",
                "early_stopping_window",
            ]:
                # Integer parameters
                return sorted(
                    list(
                        set(
                            [
                                random.randint(int(min_val), int(max_val))
                                for _ in range(self.num_samples)
                            ]
                        )
                    )
                )
            else:
                # Continuous parameters
                return sorted(
                    list(
                        set(
                            [
                                random.uniform(min_val, max_val)
                                for _ in range(self.num_samples)
                            ]
                        )
                    )
                )
        else:
            # Single fixed value
            return [param_values]

    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all possible parameter combinations from the grid."""
        sampled_params = {}

        for param_name, param_values in self.param_grid.items():
            sampled_params[param_name] = self._sample_parameter_space(
                param_name, param_values
            )

        # Generate all combinations of the sampled parameters
        param_names = sampled_params.keys()
        value_combinations = product(*sampled_params.values())

        return [dict(zip(param_names, combo)) for combo in value_combinations]

    def evaluate_parameter_set(
        self, params: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
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

            _, makespan, exec_time = processor.run(
                num_particles=params["num_particles"],
                max_iter=self.max_iter,
                w=params["w"],
                c1=params["c1"],
                c2=params["c2"],
                mutation_rate=params.get("mutation_rate", 0.0),
                max_stagnation=params.get("max_stagnation", 0),
                early_stopping_window=params.get("early_stopping_window", 0),
                improvement_threshold=params.get("improvement_threshold", 0.0),
            )

            makespans.append(makespan)
            exec_times.append(exec_time)
            individual_results[filename] = {
                "makespan": makespan,
                "exec_time": exec_time,
            }

        return mean(makespans), mean(exec_times), individual_results

    def run_search(self) -> Dict[str, Any]:
        """Execute the grid search across all parameter combinations."""
        param_combinations = self.generate_parameter_combinations()
        total_combinations = len(param_combinations)

        print(
            f"Starting grid search with {total_combinations} parameter combinations "
            f"across {len(self.dataset_files)} instances"
        )

        best_params = None
        best_avg_makespan = float("inf")
        best_avg_exec_time = float("inf")
        best_individual_results = {}

        for i, params in enumerate(param_combinations, 1):
            print(f"\nEvaluating combination {i}/{total_combinations}: {params}")

            avg_makespan, avg_exec_time, individual_results = (
                self.evaluate_parameter_set(params)
            )

            self._record_search_history(
                params, avg_makespan, avg_exec_time, individual_results
            )

            if self._is_better_solution(
                avg_makespan, avg_exec_time, best_avg_makespan, best_avg_exec_time
            ):
                best_params = params
                best_avg_makespan = avg_makespan
                best_avg_exec_time = avg_exec_time
                best_individual_results = individual_results

                print(f"New best parameters found! Avg makespan: {avg_makespan:.2f}")
                self._save_best_params(
                    best_params,
                    best_avg_makespan,
                    best_avg_exec_time,
                    best_individual_results,
                )

        self._save_search_history()
        return self._compile_final_results(
            best_params, best_avg_makespan, best_avg_exec_time, best_individual_results
        )

    def _is_better_solution(
        self,
        current_makespan: float,
        current_exec_time: float,
        best_makespan: float,
        best_exec_time: float,
    ) -> bool:
        """
        Determine if the current solution is better than the best found so far.

        Args:
            current_makespan: Makespan of current solution
            current_exec_time: Execution time of current solution
            best_makespan: Best makespan found so far
            best_exec_time: Execution time of best solution

        Returns:
            True if current solution is better, False otherwise
        """
        # Primary criterion: makespan, secondary: execution time
        if current_makespan < best_makespan:
            return True
        elif current_makespan == best_makespan and current_exec_time < best_exec_time:
            return True
        return False

    def _record_search_history(
        self,
        params: Dict[str, Any],
        avg_makespan: float,
        avg_exec_time: float,
        individual_results: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Record the results of a parameter evaluation in the search history.

        Args:
            params: Parameters used in this evaluation
            avg_makespan: Average makespan across all instances
            avg_exec_time: Average execution time across all instances
            individual_results: Results for each individual instance
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "parameters": json.dumps(params),
            "avg_makespan": avg_makespan,
            "avg_exec_time": avg_exec_time,
            **params,  # Flatten parameters into the record
        }

        # Add individual instance results
        for filename, results in individual_results.items():
            record[f"{filename}_makespan"] = results["makespan"]
            record[f"{filename}_exec_time"] = results["exec_time"]

        self.search_history.append(record)

    def _save_search_history(self) -> None:
        """Save the complete search history to a CSV file."""
        if not self.search_history:
            return

        try:
            # Get all possible fieldnames from the records
            fieldnames = set()
            for record in self.search_history:
                fieldnames.update(record.keys())

            # Ensure consistent order of fields
            fieldnames = sorted(fieldnames)

            with open(self.history_output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.search_history)

            print(f"\nSearch history saved to {self.history_output_file}")
        except Exception as e:
            print(f"Error saving search history: {str(e)}")

    def _save_best_params(
        self,
        best_params: Dict[str, Any],
        best_avg_makespan: float,
        best_avg_exec_time: float,
        best_individual_results: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Save the best parameters and their results to a JSON file.

        Args:
            best_params: Best parameters found
            best_avg_makespan: Average makespan achieved by best parameters
            best_avg_exec_time: Average execution time of best parameters
            best_individual_results: Results for each instance with best parameters
        """
        if not best_params:
            return

        result = {
            "best_parameters": best_params,
            "average_makespan": best_avg_makespan,
            "average_execution_time": best_avg_exec_time,
            "individual_results": best_individual_results,
            "timestamp": datetime.now().isoformat(),
            "search_config": {
                "max_iter": self.max_iter,
                "num_samples": self.num_samples,
                "dataset_folder": self.dataset_folder,
            },
        }

        try:
            with open(self.params_output_file, "w") as f:
                json.dump(result, f, indent=4)

            print(f"Best parameters saved to {self.params_output_file}")
        except Exception as e:
            print(f"Error saving best parameters: {str(e)}")

    def _compile_final_results(
        self,
        best_params: Dict[str, Any],
        best_avg_makespan: float,
        best_avg_exec_time: float,
        best_individual_results: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Compile the final results of the grid search.

        Args:
            best_params: Best parameters found
            best_avg_makespan: Average makespan achieved by best parameters
            best_avg_exec_time: Average execution time of best parameters
            best_individual_results: Results for each instance with best parameters

        Returns:
            Dictionary containing all results and metadata
        """
        return {
            "best_parameters": best_params,
            "average_makespan": best_avg_makespan,
            "average_execution_time": best_avg_exec_time,
            "individual_results": best_individual_results,
            "search_history_file": self.history_output_file,
            "parameters_file": self.params_output_file,
            "total_evaluations": len(self.search_history),
            "dataset_files": self.dataset_files,
            "timestamp": datetime.now().isoformat(),
        }
