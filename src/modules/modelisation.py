from typing import List, Tuple


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
        self.num_machines = len(machines_matrix[0])
        self.jobs = [
            Job(j, machines_matrix[j], times_matrix[j]) for j in range(self.num_jobs)
        ]
        self.schedule = {}  # Stores start & end times per machine
        self.job_machine_dict = { # Maps job IDs to their machine IDs : Speeds up access during scheduling (no need to recompute machine assignments repeatedly).
            job_idx: [op.machine - 1 for op in self.jobs[job_idx].operations] #Creates a lookup table where, for each job (job_idx), we store the machine IDs (0-based index) for its operations because  Machine IDs are 1-indexed in the input
            for job_idx in range(self.num_jobs)
        }
        self.initialize_schedule()

    def initialize_schedule(self):
        """Creates an empty schedule for all machines."""
        self.schedule = {m: [] for m in range(1, self.num_machines + 1)} # self.schedule is populated with keys for each machine (e.g., {1: [], 2: [], 3: []} for 3 machines).

    def __repr__(self):
        return f"JSSP with {self.num_jobs} Jobs and {self.num_machines} Machines"

    def evaluate_schedule(self, operation_sequence: List[Tuple[int, int]]) -> int:
        """Evaluates a schedule and returns the makespan."""
        for job in self.jobs:
            for op in job.operations:
                op.start_time = None
                op.end_time = None
            job.current_operation_index = 0

        self.initialize_schedule()
        job_times = [0] * self.num_jobs # Tracks the completion time of each job (initially 0).
        machine_times = {m: 0 for m in self.schedule.keys()} # Tracks when each machine is free

        for job_idx, op_idx in operation_sequence:
            job = self.jobs[job_idx]
            op = job.operations[op_idx]
            start_time = max(job_times[job_idx], machine_times[op.machine]) # Start when job and machine are free 
            end_time = start_time + op.processing_time
            op.start_time = start_time
            op.end_time = end_time
            job_times[job_idx] = end_time
            machine_times[op.machine] = end_time

        return max(job_times)
