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
    
machines_matrix = [
    [4, 12, 15],  # Temps de traitement des machines pour Job 0
    [6, 1, 4]     # Temps de traitement des machines pour Job 1
]

times_matrix = [
    [25, 75, 75],  # Durées des opérations pour Job 0
    [67, 5, 11]    # Durées des opérations pour Job 1
]

# Creating a JSSP instance
jssp = JSSP(machines_matrix, times_matrix)
print(jssp)
print(jssp.jobs)
