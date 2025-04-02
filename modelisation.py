import numpy as np

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
        self.jobs = [Job(j, machines_matrix[j], times_matrix[j]) for j in range(self.num_jobs)]
        self.schedule = {}  # Stores start & end times per machine

    def initialize_schedule(self):
        """Creates an empty schedule for all machines."""
        self.schedule = {m: [] for m in range(1, self.num_machines + 1)}

    def __repr__(self):
        return f"JSSP with {self.num_jobs} Jobs and {self.num_machines} Machines"

# Example: Loading dataset
machines_matrix = [
    [4, 12, 15, 2, 11, 3, 5, 8, 1, 13, 6, 10, 7, 14, 9],
    [6, 1, 4, 9, 5, 2, 13, 15, 7, 8, 11, 3, 10, 14, 12]
]
times_matrix = [
    [25, 75, 75, 76, 38, 62, 38, 59, 14, 13, 46, 31, 57, 92, 3],
    [67, 5, 11, 11, 40, 34, 77, 42, 35, 96, 22, 55, 21, 29, 16]
]

# Creating a JSSP instance
jssp = JSSP(machines_matrix, times_matrix)
print(jssp)
print(jssp.jobs)
