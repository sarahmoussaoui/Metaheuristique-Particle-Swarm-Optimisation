import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from copy import deepcopy


class Operation:
    def __init__(self, machine: int, processing_time: int):
        self.machine = machine
        self.processing_time = processing_time
        self.start_time = None
        self.end_time = None

    def __repr__(self):
        return f"(M{self.machine}, T{self.processing_time})"


class Job:
    def __init__(self, job_id: int, machines: list, times: list):
        self.job_id = job_id
        self.operations = [Operation(m, t) for m, t in zip(machines, times)]
        self.current_operation_index = 0

    def __repr__(self):
        return f"Job {self.job_id}: {self.operations}"


class JSSP:
    def __init__(self, machines_matrix, times_matrix):
        self.num_jobs = len(machines_matrix)
        self.num_machines = len(machines_matrix[0])
        self.jobs = [
            Job(j, machines_matrix[j], times_matrix[j]) for j in range(self.num_jobs)
        ]
        self.schedule = {}

    def reset(self):
        for job in self.jobs:
            job.current_operation_index = 0
        self.schedule = {}

    def generate_gantt_chart(self, solution):
        self.reset()
        self.schedule = {m: [] for m in range(1, self.num_machines + 1)}
        machine_available = {m: 0 for m in range(1, self.num_machines + 1)}
        job_available = {j.job_id: 0 for j in self.jobs}

        # Create operation lookup table
        op_lookup = {}
        for job in self.jobs:
            for op in job.operations:
                op_lookup[(job.job_id, op.machine, op.processing_time)] = op

        for job_id, machine, processing_time in solution:
            op = op_lookup[(job_id, machine, processing_time)]
            job = self.jobs[job_id]

            # Verify this is the next operation for the job
            if job.operations[job.current_operation_index] != op:
                continue

            start_time = max(machine_available[machine], job_available[job_id])
            end_time = start_time + processing_time

            op.start_time = start_time
            op.end_time = end_time
            machine_available[machine] = end_time
            job_available[job_id] = end_time
            job.current_operation_index += 1

            self.schedule[machine].append(
                {
                    "job_id": job_id,
                    "start": start_time,
                    "end": end_time,
                    "operation": op,
                }
            )

        return self.schedule

    def calculate_makespan(self, solution=None):
        if solution is not None:
            self.generate_gantt_chart(solution)

        if not self.schedule:
            return float("inf")

        return max(op["end"] for ops in self.schedule.values() for op in ops)

    def generate_random_solution(self):
        # Create a list of all operations with their job and position
        ops_info = []
        for job in self.jobs:
            for op_idx, operation in enumerate(job.operations):
                ops_info.append((job.job_id, op_idx, operation))

        # Shuffle while maintaining operation order within jobs
        random.shuffle(ops_info)

        # Reconstruct solution respecting operation sequence
        solution = []
        op_counters = {job.job_id: 0 for job in self.jobs}

        while len(solution) < sum(len(job.operations) for job in self.jobs):
            for job_id, op_idx, operation in ops_info:
                if op_counters[job_id] == op_idx:
                    solution.append((job_id, operation))
                    op_counters[job_id] += 1

        return solution

    def print_gantt_chart(self, solution):
        """Prints the Gantt chart in a readable format."""
        print("\nGantt Chart:")
        for machine in sorted(self.schedule.keys()):
            print(f"Machine {machine}:")
            for op in sorted(self.schedule[machine], key=lambda x: x["start"]):
                print(
                    f"  Job {op['job_id']}: {op['start']}-{op['end']} (Operation {op['operation']})"
                )
        # Shuffle while maintaining operation order within jobs
        job_op_indices = {job.job_id: 0 for job in self.jobs}
        shuffled = []

        while len(shuffled) < len(solution):
            for job in self.jobs:
                if job_op_indices[job.job_id] < len(job.operations):
                    op = job.operations[job_op_indices[job.job_id]]
                    if (job.job_id, op.machine, op.processing_time) in solution:
                        shuffled.append((job.job_id, op.machine, op.processing_time))
                        job_op_indices[job.job_id] += 1

        return shuffled


class PriorityPSOSolver:
    def __init__(self, jssp, num_particles=30, max_iter=200, w=0.8, c1=1.2, c2=1.2):
        self.jssp = jssp
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # inertia
        self.c1 = c1  # cognitive
        self.c2 = c2  # social
        self.best_fitness_history = []

        # Each operation gets a priority value
        self.num_operations = sum(len(job.operations) for job in jssp.jobs)
        self.global_best = None
        self.global_best_fitness = float("inf")

    def initialize_particles(self):
        particles = []
        for _ in range(self.num_particles):
            # Random priorities between 0 and 1
            priorities = np.random.rand(self.num_operations)
            solution = self.decode(priorities)
            fitness = self.jssp.calculate_makespan(solution)

            particles.append(
                {
                    "position": priorities,
                    "velocity": np.random.rand(self.num_operations) * 0.1,
                    "best_position": priorities.copy(),
                    "best_fitness": fitness,
                    "fitness": fitness,
                }
            )

            if fitness < self.global_best_fitness:
                self.global_best = priorities.copy()
                self.global_best_fitness = fitness
        return particles

    def decode(self, priorities):
        """Convert priority values to a schedule while respecting operation order within jobs"""
        # Create operation records with job info and priority
        ops = []
        idx = 0
        for job in self.jssp.jobs:
            for op_idx, op in enumerate(job.operations):
                ops.append(
                    {
                        "job_id": job.job_id,
                        "op": op,
                        "op_idx": op_idx,
                        "priority": priorities[idx],
                    }
                )
                idx += 1

        # Sort all operations by priority (highest first)
        ops_sorted = sorted(ops, key=lambda x: -x["priority"])

        # Build solution respecting operation order within jobs
        solution = []
        job_next_op = {
            job.job_id: 0 for job in self.jssp.jobs
        }  # tracks next operation needed for each job

        # We may need multiple passes to ensure all operations are scheduled
        while len(solution) < self.num_operations:
            for item in ops_sorted:
                job_id = item["job_id"]
                op_idx = item["op_idx"]
                op = item["op"]

                # Only add if it's the next operation needed for this job
                if job_next_op[job_id] == op_idx:
                    solution.append((job_id, op.machine, op.processing_time))
                    job_next_op[job_id] += 1

                    # Break to restart the process with updated job_next_op
                    break

        return solution

    def solve(self):
        particles = self.initialize_particles()

        for iteration in range(self.max_iter):
            for particle in particles:
                # Update velocity (standard PSO equations)
                r1, r2 = np.random.rand(2)
                cognitive = (
                    self.c1 * r1 * (particle["best_position"] - particle["position"])
                )
                social = self.c2 * r2 * (self.global_best - particle["position"])
                particle["velocity"] = (
                    self.w * particle["velocity"] + cognitive + social
                )

                # Update position
                particle["position"] += particle["velocity"]

                # Evaluate new position
                solution = self.decode(particle["position"])
                fitness = self.jssp.calculate_makespan(solution)
                particle["fitness"] = fitness

                # Update personal best
                if fitness < particle["best_fitness"]:
                    particle["best_position"] = particle["position"].copy()
                    particle["best_fitness"] = fitness

                    # Update global best
                    if fitness < self.global_best_fitness:
                        self.global_best = particle["position"].copy()
                        self.global_best_fitness = fitness

            self.best_fitness_history.append(self.global_best_fitness)
            if (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}, Best Makespan: {self.global_best_fitness}"
                )

        best_solution = self.decode(self.global_best)
        return best_solution, self.global_best_fitness

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, "b-", linewidth=2)
        plt.xlabel("Iteration")
        plt.ylabel("Best Makespan")
        plt.title("PSO Convergence")
        plt.grid(True)
        plt.show()


def plot_gantt_chart(jssp):
    fig, ax = plt.subplots(figsize=(12, 6))
    colormap = plt.colormaps["tab20"]
    colors = [colormap(i) for i in np.linspace(0, 1, jssp.num_jobs)]
    max_time = max(
        max((op["end"] for op in ops), default=0) for ops in jssp.schedule.values()
    )

    ax.set_xlim(0, max_time + 10)
    ax.set_ylim(0.5, jssp.num_machines + 0.5)
    ax.set_yticks(range(1, jssp.num_machines + 1))
    ax.set_yticklabels([f"Machine {m}" for m in range(1, jssp.num_machines + 1)])
    ax.set_xlabel("Time")
    ax.set_title("Job Shop Schedule Gantt Chart")

    for machine, ops in jssp.schedule.items():
        for op in ops:
            job_id = op["job_id"]
            start = op["start"]
            duration = op["end"] - op["start"]

            rect = patches.Rectangle(
                (start, machine - 0.4),
                duration,
                0.8,
                facecolor=colors[job_id % len(colors)],
                edgecolor="black",
                label=f"Job {job_id}",
            )
            ax.add_patch(rect)

            ax.text(
                start + duration / 2,
                machine,
                f"J{job_id}\n{duration}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if np.mean(colors[job_id % len(colors)][:3]) < 0.5
                    else "black"
                ),
                fontsize=8,
            )

    handles, labels = [], []
    for machine, ops in jssp.schedule.items():
        for op in ops:
            label = f'Job {op["job_id"]}'
            if label not in labels:
                handles.append(
                    patches.Patch(color=colors[op["job_id"] % len(colors)], label=label)
                )
                labels.append(label)

    ax.legend(handles=handles, title="Jobs", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.grid(True, axis="x")
    plt.show()


if __name__ == "__main__":
    # Benchmark instance FT06 (3 jobs, 6 machines)
    times_matrix = [
        [25, 75, 75, 76, 38, 62, 38, 59, 14, 13, 46, 31, 57, 92, 3],
        [67, 5, 11, 11, 40, 34, 77, 42, 35, 96, 22, 55, 21, 29, 16],
        [22, 98, 8, 35, 59, 31, 13, 46, 52, 22, 18, 19, 64, 29, 70],
        [99, 42, 2, 35, 11, 92, 88, 97, 21, 56, 17, 43, 27, 19, 23],
        [50, 5, 59, 71, 47, 39, 82, 35, 12, 2, 39, 42, 52, 65, 35],
        [48, 57, 5, 2, 60, 64, 86, 3, 51, 26, 34, 39, 45, 63, 54],
        [40, 43, 50, 71, 46, 99, 67, 34, 6, 95, 67, 54, 29, 30, 60],
        [59, 3, 85, 6, 46, 49, 5, 82, 18, 71, 48, 79, 62, 65, 76],
        [65, 55, 81, 15, 32, 52, 97, 69, 82, 89, 69, 87, 22, 71, 63],
        [70, 74, 52, 94, 14, 81, 24, 14, 32, 39, 67, 59, 18, 77, 50],
        [18, 6, 96, 53, 35, 99, 39, 18, 14, 90, 64, 81, 89, 48, 80],
        [44, 75, 12, 13, 74, 59, 71, 75, 30, 93, 26, 30, 84, 91, 93],
        [39, 56, 13, 29, 55, 69, 26, 7, 55, 48, 22, 46, 50, 96, 17],
        [57, 14, 8, 13, 95, 53, 78, 24, 92, 90, 68, 87, 43, 75, 94],
        [93, 92, 18, 28, 27, 40, 56, 83, 51, 15, 97, 48, 53, 78, 39],
        [47, 34, 42, 28, 11, 11, 30, 14, 10, 4, 20, 92, 19, 59, 28],
        [69, 82, 64, 40, 27, 82, 27, 43, 56, 17, 18, 20, 98, 43, 68],
        [84, 26, 87, 61, 95, 23, 88, 89, 49, 84, 12, 51, 3, 44, 20],
        [43, 54, 18, 72, 70, 28, 20, 22, 59, 36, 85, 13, 73, 29, 45],
        [7, 97, 4, 22, 74, 45, 62, 95, 66, 14, 40, 23, 79, 34, 8],
    ]

    machines_matrix = [
        [4, 12, 15, 2, 11, 3, 5, 8, 1, 13, 6, 10, 7, 14, 9],
        [6, 1, 4, 9, 5, 2, 13, 15, 7, 8, 11, 3, 10, 14, 12],
        [3, 4, 15, 1, 10, 13, 6, 5, 8, 11, 9, 12, 14, 2, 7],
        [9, 11, 2, 14, 4, 5, 15, 10, 3, 6, 12, 8, 1, 7, 13],
        [15, 9, 2, 3, 11, 10, 13, 5, 7, 6, 1, 14, 4, 12, 8],
        [4, 11, 2, 6, 7, 1, 9, 8, 12, 14, 3, 15, 13, 10, 5],
        [3, 11, 2, 13, 9, 1, 8, 7, 15, 14, 5, 4, 6, 10, 12],
        [2, 1, 3, 5, 8, 14, 12, 4, 13, 6, 7, 15, 10, 9, 11],
        [5, 6, 10, 11, 8, 7, 3, 2, 13, 4, 14, 1, 9, 15, 12],
        [2, 5, 4, 11, 15, 1, 7, 14, 12, 9, 6, 13, 8, 10, 3],
        [4, 11, 2, 1, 10, 9, 15, 7, 5, 8, 3, 13, 6, 12, 14],
        [3, 8, 7, 9, 4, 6, 15, 5, 2, 1, 10, 11, 14, 12, 13],
        [1, 8, 15, 9, 13, 11, 10, 4, 7, 2, 5, 3, 12, 14, 6],
        [13, 4, 10, 5, 2, 1, 11, 7, 6, 3, 15, 14, 8, 9, 12],
        [4, 15, 7, 6, 14, 10, 2, 1, 13, 8, 3, 5, 11, 9, 12],
        [6, 15, 7, 13, 9, 3, 5, 10, 12, 14, 4, 2, 8, 1, 11],
        [4, 8, 11, 15, 1, 9, 2, 12, 6, 14, 5, 13, 7, 10, 3],
        [11, 9, 3, 12, 14, 7, 15, 4, 10, 8, 5, 6, 13, 1, 2],
        [4, 3, 13, 14, 2, 7, 15, 6, 5, 9, 10, 12, 1, 11, 8],
        [12, 15, 6, 7, 11, 10, 14, 2, 5, 9, 1, 4, 13, 3, 8],
    ]

    jssp = JSSP(machines_matrix, times_matrix)
    pso = PriorityPSOSolver(jssp, num_particles=30, max_iter=200, w=0.8, c1=1.2, c2=1.2)
    best_solution, best_makespan = pso.solve()

    print(f"\nBest makespan found: {best_makespan}")
    pso.plot_convergence()

    jssp.generate_gantt_chart(best_solution)
    jssp.print_gantt_chart(best_solution)
    plot_gantt_chart(jssp)
