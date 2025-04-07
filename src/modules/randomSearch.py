import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Tuple, Union
import random
from statistics import mean
from src.modules.jsspProcessor import JSSPProcessor


class PSORandomSearch:
    def __init__(
        self,
        max_iter: int,
        dataset_folder: str,
        output_prefix: str = "pso_search",
        num_samples: int = 200,
        seed: int = None,
    ):
        """
        Initialize the random search with dataset folder and output files.

        Args:
            max_iter: Maximum iterations for PSO
            dataset_folder: Path to folder containing JSSP datasets
            output_prefix: Prefix for output files (will generate .json and .csv)
            num_samples: Number of random parameter combinations to try
            seed: Random seed for reproducibility
        """
        self.dataset_folder = dataset_folder
        self.num_samples = num_samples
        self.max_iter = max_iter
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # Output files
        self.json_output = f"{output_prefix}_best.json"
        self.csv_output = f"{output_prefix}_history.csv"
        self.best_results: Dict[str, Any] = {}
        self.search_history: List[Dict[str, Any]] = []

        # Parameter specifications (can be ranges or exact values)
        self.param_specs = {
            "num_particles": (20, 50),  # Range
            "w": [0.4, 0.6, 0.8],  # List of exact values
            "c1": (1.0, 2.0),  # Range
            "c2": (1.0, 2.0),  # Range
            "mutation_rate": 0.1,  # Exact value
            "max_stagnation": (10, 30),  # Range
            "early_stopping_window": [10, 20],  # List of exact values
            "improvement_threshold": 0.01,  # Exact value
        }

        # Get all dataset files
        self.dataset_files = [
            os.path.join(dataset_folder, f)
            for f in os.listdir(dataset_folder)
            if f.endswith(".txt")
        ]

    def set_parameter_specs(
        self, specs: Dict[str, Union[Tuple, List, float, int]]
    ) -> None:
        """Set custom parameter specifications for the search.

        Args:
            specs: Dictionary where values can be:
                   - Tuple (min, max) for random range
                   - List of exact values to choose from
                   - Single value (int/float) for fixed parameter
        """
        self.param_specs = specs

    def generate_parameters(self) -> Dict[str, Any]:
        """Generate a parameter combination according to specifications."""
        params = {}
        for param, spec in self.param_specs.items():
            if isinstance(spec, tuple):  # Range
                if isinstance(spec[0], int) and isinstance(spec[1], int):
                    params[param] = random.randint(spec[0], spec[1])
                else:
                    params[param] = random.uniform(spec[0], spec[1])
            elif isinstance(spec, list):  # List of options
                params[param] = random.choice(spec)
            else:  # Fixed value
                params[param] = spec
        return params

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

            _, makespan, exec_time = processor.run(
                num_particles=params["num_particles"],
                max_iter=self.max_iter,
                w=params["w"],
                c1=params["c1"],
                c2=params["c2"],
                mutation_rate=params["mutation_rate"],
                max_stagnation=params["max_stagnation"],
                early_stopping_window=params["early_stopping_window"],
                improvement_threshold=params["improvement_threshold"],
            )
            makespans.append(makespan)
            exec_times.append(exec_time)
            individual_results[filename] = {
                "makespan": makespan,
                "exec_time": exec_time,
            }

        return mean(makespans), mean(exec_times), individual_results

    def run_search(self) -> Dict[str, Any]:
        """Execute the parameter search across combinations."""
        best_avg_makespan = float("inf")
        best_params = None
        best_avg_exec_time = float("inf")
        best_individual_results = {}

        print(
            f"Starting parameter search with {self.num_samples} samples "
            f"across {len(self.dataset_files)} instances"
        )

        # Prepare CSV file
        with open(self.csv_output, "w", newline="") as csvfile:
            fieldnames = (
                ["sample_num", "timestamp"]
                + list(self.param_specs.keys())
                + ["avg_makespan", "avg_exec_time"]
            )
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(1, self.num_samples + 1):
                params = self.generate_parameters()
                print(f"\nEvaluating sample {i}/{self.num_samples}: {params}")

                avg_makespan, avg_exec_time, individual_results = (
                    self.evaluate_parameter_set(params)
                )

                # Record in history
                record = {
                    "sample_num": i,
                    "timestamp": datetime.now().isoformat(),
                    **params,
                    "avg_makespan": avg_makespan,
                    "avg_exec_time": avg_exec_time,
                }
                self.search_history.append(record)
                writer.writerow(record)

                # Check if this is the best so far
                if avg_makespan < best_avg_makespan or (
                    avg_makespan == best_avg_makespan
                    and avg_exec_time < best_avg_exec_time
                ):
                    best_avg_makespan = avg_makespan
                    best_avg_exec_time = avg_exec_time
                    best_params = params
                    best_individual_results = individual_results

                    print(
                        f"New best parameters found! Avg makespan: {avg_makespan:.2f}"
                    )
                    self.save_current_best(best_params, best_avg_makespan)

        # Save final results
        self.best_results = {
            "best_parameters": best_params,
            "best_avg_makespan": best_avg_makespan,
            "best_avg_exec_time": best_avg_exec_time,
            "individual_results": best_individual_results,
            "search_samples": self.num_samples,
            "dataset_files": [os.path.basename(f) for f in self.dataset_files],
            "random_seed": self.seed,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self.json_output, "w") as f:
            json.dump(self.best_results, f, indent=2)

        return self.best_results

    def save_current_best(self, params: Dict[str, Any], avg_makespan: float) -> None:
        """Save current best results during the search."""
        current_best = {
            "current_best_parameters": params,
            "current_avg_makespan": avg_makespan,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.json_output, "w") as f:
            json.dump(current_best, f, indent=2)

    def get_search_history(self) -> List[Dict[str, Any]]:
        """Return the complete search history."""
        return self.search_history
