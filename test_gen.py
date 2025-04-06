from src.modules.modelisation import JSSP
from src.modules.datasetParser import DatasetParser
from src.modules.visualisation import ScheduleVisualizer
from src.modules.genetic import GeneticAlgorithm
import os
import time
import csv


class JSSPProcessor:
    def __init__(
        self,
        dataset_path,
        plot: bool = True,
        output_base="output_gen",
    ):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.output_dir = os.path.join(output_base, self.dataset_name)
        self.log_file = os.path.join(self.output_dir, "results.csv")
        self.plot = plot

    def run(self, population_size=20, generations=50, mutation_rate=0.1):
        with open(self.dataset_path, "r") as file:
            dataset_str = file.read()

        num_jobs, num_machines, upper_bound, lower_bound, times, machines = (
            DatasetParser.parse(dataset_str)
        )
        os.makedirs(self.output_dir, exist_ok=True)

        jssp = JSSP(machines, times)

        optimizer = GeneticAlgorithm(jssp)
        start_time = time.time()
        best_schedule, best_makespan = optimizer.evolve(
            mutation_rate=mutation_rate,
            population_size=population_size,
            generations=generations,
        )
        exec_time = time.time() - start_time

        if self.plot:
            ScheduleVisualizer.plot_convergence(
                optimizer.iteration_history,
                optimizer.makespan_history,
                upper_bound=upper_bound,
                save_folder=self.output_dir,
            )
            ScheduleVisualizer.plot_gantt_chart(jssp, save_folder=self.output_dir)

            self._log_results(best_makespan, exec_time)

        print(
            f"[{self.dataset_name}] Makespan: {best_makespan} | Time: {exec_time:.2f}s"
        )
        return best_schedule, best_makespan, exec_time

    def _log_results(self, best_makespan, exec_time):
        csv_exists = os.path.exists(self.log_file)
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not csv_exists:
                writer.writerow(["Dataset", "Best Makespan", "Execution Time (s)"])
            writer.writerow([self.dataset_name, best_makespan, f"{exec_time:.4f}"])


if __name__ == "__main__":
    # Example usage
    processor = JSSPProcessor(
        dataset_path="./src/data/processed/data_20j_15m/data_20j_15m_1.txt"
    )
    processor.run(population_size=50, generations=100, mutation_rate=0.05)
