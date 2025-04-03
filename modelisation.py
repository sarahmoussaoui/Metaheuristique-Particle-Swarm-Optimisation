import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Operation:
    """Represents an operation in a job."""

    def __init__(self, machine: int, processing_time: int):
        self.machine = machine  # Machine ID
        self.processing_time = processing_time  # Time required
        self.start_time = None
        self.end_time = None

    def __repr__(self):
        return f"(M{self.machine}, T{self.processing_time})"


class Job:
    """Represents a job consisting of multiple operations."""

    def __init__(self, job_id: int, machines: list, times: list):
        self.job_id = job_id
        self.operations = [Operation(m, t) for m, t in zip(machines, times)]
        self.current_operation_index = 0  # Tracks execution progress

    def __repr__(self):
        return f"Job {self.job_id}: {self.operations}"


class JSSP:
    """Manages the entire Job Shop Scheduling Problem."""

    def __init__(self, machines_matrix, times_matrix):
        self.num_jobs = len(machines_matrix)
        self.num_machines = len(
            machines_matrix[0]
        )  # what if diffrent number of machines for jobs ?
        self.jobs = [
            Job(j, machines_matrix[j], times_matrix[j]) for j in range(self.num_jobs)
        ]
        self.schedule = {}  # Stores start & end times per machine

    def initialize_schedule(self):
        """Creates an empty schedule for all machines."""
        self.schedule = {m: [] for m in range(1, self.num_machines + 1)}
        self.machine_availability = {m: 0 for m in range(1, self.num_machines + 1)}
        self.job_progress = {j.job_id: 0 for j in self.jobs}

    def generate_gantt_chart(self, solution):
        """Generates a Gantt chart from a solution."""
        self.initialize_schedule()

        # Process operations in the order of the solution
        for job_id, operation in solution:
            job = self.jobs[job_id]

            # Check if this is the next operation for the job
            if (
                job.current_operation_index >= len(job.operations)
                or operation != job.operations[job.current_operation_index]
            ):
                continue

            machine = operation.machine
            processing_time = operation.processing_time

            # Determine start time
            start_time = max(
                self.machine_availability[machine], self.job_progress[job_id]
            )
            end_time = start_time + processing_time

            # Update operation times
            operation.start_time = start_time
            operation.end_time = end_time

            # Update tracking variables
            self.machine_availability[machine] = end_time
            self.job_progress[job_id] = end_time
            job.current_operation_index += 1

            # Add to schedule
            self.schedule[machine].append(
                {
                    "job_id": job_id,
                    "start": start_time,
                    "end": end_time,
                    "operation": operation,
                }
            )

        return self.schedule

    def calculate_makespan(self):
        """Calculates the makespan (total completion time) of the current schedule."""
        if not self.schedule:
            return 0

        max_end_time = 0
        for machine_ops in self.schedule.values():
            for op in machine_ops:
                if op["end"] > max_end_time:
                    max_end_time = op["end"]
        return max_end_time

    def generate_random_solution(self):
        """Generates a valid random solution respecting operation order."""
        # Create a list of all operations with their job and position
        ops_info = []
        for job in self.jobs:
            for op_idx, operation in enumerate(job.operations):
                ops_info.append((job.job_id, op_idx, operation))

        # Shuffle while maintaining operation order within jobs
        random.shuffle(ops_info)

        # Reconstruct solution in shuffled order but respecting operation sequence
        solution = []
        op_counters = {job.job_id: 0 for job in self.jobs}

        while len(solution) < sum(len(job.operations) for job in self.jobs):
            for job_id, op_idx, operation in ops_info:
                if op_counters[job_id] == op_idx:
                    solution.append((job_id, operation))
                    op_counters[job_id] += 1

        return solution

    def print_gantt_chart(self):
        """Prints the Gantt chart in a readable format."""
        print("\nGantt Chart:")
        for machine in sorted(self.schedule.keys()):
            print(f"Machine {machine}:")
            for op in sorted(self.schedule[machine], key=lambda x: x["start"]):
                print(
                    f"  Job {op['job_id']}: {op['start']}-{op['end']} (Operation {op['operation']})"
                )

    def __repr__(self):
        return f"JSSP with {self.num_jobs} Jobs and {self.num_machines} Machines"


def plot_gantt_chart(jssp):
    """Plots the Gantt chart using matplotlib."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get colormap (updated method)
    colormap = plt.colormaps["tab20"]
    colors = [colormap(i) for i in np.linspace(0, 1, jssp.num_jobs)]

    # Find maximum time
    max_time = max(
        max((op["end"] for op in ops), default=0) for ops in jssp.schedule.values()
    )

    # Set up plot
    ax.set_xlim(0, max_time + 10)
    ax.set_ylim(0.5, jssp.num_machines + 0.5)
    ax.set_yticks(range(1, jssp.num_machines + 1))
    ax.set_yticklabels([f"Machine {m}" for m in range(1, jssp.num_machines + 1)])
    ax.set_xlabel("Time")
    ax.set_title("Job Shop Schedule Gantt Chart")

    # Plot each operation
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

            # Add text
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

    # Create legend without duplicates
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

    def __repr__(self):
        return f"JSSP with {self.num_jobs} Jobs and {self.num_machines} Machines"


# Example: Loading dataset
machines_matrix = [
    [1, 4, 3, 2],
    [2, 1, 3, 4],
    [2, 1, 4, 3],
]
times_matrix = [
    [25, 75, 75, 76],
    [67, 5, 11, 11],
    [7, 5, 40, 18],
]

# Creating a JSSP instance


jssp = JSSP(machines_matrix, times_matrix)
print(jssp)
print("jobs : \n", jssp.jobs, "\n\n")
random_solution = jssp.generate_random_solution()
print("random_solution : \n", random_solution, "\n\n")
gantt = jssp.generate_gantt_chart(random_solution)
jssp.print_gantt_chart()
plot_gantt_chart(jssp)  # This will display the visual Gantt chart
print(jssp.calculate_makespan())
