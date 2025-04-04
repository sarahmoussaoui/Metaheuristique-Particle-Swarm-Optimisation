import os
import time
import csv
import matplotlib.pyplot as plt
from src.modules.modelisation import JSSP
from src.modules.visualisation import ScheduleVisualizer
from src.modules.pso import PSOOptimizer
from src.modules.datasetParser import DatasetParser


class JSSPProcessor:
    def __init__(self, dataset_path, output_base="output"):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.output_dir = os.path.join(output_base, self.dataset_name)
        self.log_file = os.path.join(self.output_dir, "results.csv")

    def run(self):
        with open(self.dataset_path, "r") as file:
            dataset_str = file.read()

        num_jobs, num_machines, times, machines = DatasetParser.parse(dataset_str)
        os.makedirs(self.output_dir, exist_ok=True)

        jssp = JSSP(machines, times)

        optimizer = PSOOptimizer(jssp)
        start_time = time.time()
        best_schedule, best_makespan = optimizer.optimize(num_particles=20, max_iter=50)
        exec_time = time.time() - start_time

        ScheduleVisualizer.plot_convergence(
            optimizer.iteration_history,
            optimizer.makespan_history,
            save_folder=self.output_dir,
        )
        ScheduleVisualizer.plot_gantt_chart(jssp, save_folder=self.output_dir)

        self._log_results(best_makespan, exec_time)

        print(
            f"[{self.dataset_name}] Makespan: {best_makespan} | Time: {exec_time:.2f}s"
        )

    def _log_results(self, best_makespan, exec_time):
        csv_exists = os.path.exists(self.log_file)
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not csv_exists:
                writer.writerow(["Dataset", "Best Makespan", "Execution Time (s)"])
            writer.writerow([self.dataset_name, best_makespan, f"{exec_time:.4f}"])


# === MAIN ===
if __name__ == "__main__":
    dataset_folder = "src/data/"

    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            dataset_path = os.path.join(dataset_folder, filename)
            processor = JSSPProcessor(dataset_path)
            processor.run()
