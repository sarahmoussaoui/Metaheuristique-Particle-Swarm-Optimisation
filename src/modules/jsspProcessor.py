import os
import time
import csv
from src.modules.modelisation import JSSP
from src.modules.visualisation import ScheduleVisualizer
from src.modules.pso import PSOOptimizer
from src.modules.datasetParser import DatasetParser


class JSSPProcessor:
    def __init__(
        self,
        dataset_path,
        plot: bool = True,
        output_base="output",
    ):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        self.output_dir = os.path.join(output_base, self.dataset_name)
        self.log_file = os.path.join(self.output_dir, "results.csv")
        self.plot = plot

    def run(
        self,
        num_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        with open(self.dataset_path, "r") as file:
            dataset_str = file.read()

        num_jobs, num_machines, upper_bound, lower_bound, times, machines = (
            DatasetParser.parse(dataset_str)
        )
        os.makedirs(self.output_dir, exist_ok=True)

        jssp = JSSP(machines, times)

        optimizer = PSOOptimizer(jssp)
        start_time = time.time()
        best_schedule, best_makespan = optimizer.optimize(
            num_particles=num_particles,
            max_iter=max_iter,
            w=w,
            c1=c1,
            c2=c2,
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
