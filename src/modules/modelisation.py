from typing import List, Dict, Tuple

class Operation:
    """Represents an operation in a job."""

    def __init__(self, machine: int, processing_time: int):
        self.machine = machine  # Machine ID (1-based)
        self.processing_time = processing_time  # Time required
        self.start_time = None
        self.end_time = None

    def __repr__(self):
        return f"(M{self.machine}, T{self.processing_time})"

class Job:
    """Represents a job consisting of multiple operations."""

    def __init__(self, job_id: int, machines: List[int], times: List[int]):
        self.job_id = job_id
        self.operations = [Operation(m, t) for m, t in zip(machines, times)]
        self.current_operation_index = 0  # Tracks execution progress

    def __repr__(self):
        return f"Job {self.job_id}: {self.operations}"

class JSSP:
    """Manages the entire Job Shop Scheduling Problem."""

    def __init__(self, machines_matrix: List[List[int]], times_matrix: List[List[int]]):
        """
        Initialize JSSP instance.
        
        Args:
            machines_matrix: Matrix where machines_matrix[j][o] gives machine for operation o of job j
            times_matrix: Matrix where times_matrix[j][o] gives processing time for operation o of job j
        """
        self.num_jobs = len(machines_matrix)
        self.num_machines = max(max(machines) for machines in machines_matrix)  # Get max machine ID
        self.times_matrix = times_matrix  # Store processing times matrix
        self.machines_matrix = machines_matrix  # Store machines matrix
        
        # Create Job objects
        self.jobs = [
            Job(j, machines_matrix[j], times_matrix[j]) 
            for j in range(self.num_jobs)
        ]
        
        # Initialize schedule and lookup dictionaries
        self.schedule = {}  
        self.job_machine_dict = {
            job_idx: [op.machine for op in self.jobs[job_idx].operations]
            for job_idx in range(self.num_jobs)
        }
        self.initialize_schedule()

    def initialize_schedule(self):
        """Creates an empty schedule for all machines."""
        self.schedule = {m: [] for m in range(1, self.num_machines + 1)}

    def __repr__(self):
        return f"JSSP({self.num_jobs} jobs, {self.num_machines} machines)"

    def evaluate_schedule(self, operation_sequence: List[Tuple[int, int]]) -> int:
        """
        Evaluates a schedule and returns the makespan.
        
        Args:
            operation_sequence: List of (job_idx, op_idx) tuples
            
        Returns:
            int: Makespan of the schedule
        """
        # Reset tracking variables
        for job in self.jobs:
            for op in job.operations:
                op.start_time = None
                op.end_time = None
            job.current_operation_index = 0

        self.initialize_schedule()
        job_completion_times = [0] * self.num_jobs
        machine_available_times = {m: 0 for m in self.schedule.keys()}

        for job_idx, op_idx in operation_sequence:
            job = self.jobs[job_idx]
            op = job.operations[op_idx]
            
            # Operation can start when both:
            # 1. The job has finished its previous operation
            # 2. The machine is available
            start_time = max(job_completion_times[job_idx], machine_available_times[op.machine])
            end_time = start_time + op.processing_time
            
            # Update operation times
            op.start_time = start_time
            op.end_time = end_time
            
            # Update tracking variables
            job_completion_times[job_idx] = end_time
            machine_available_times[op.machine] = end_time

        return max(job_completion_times)  # The makespan is the maximum completion time

    def get_operation_processing_time(self, job_idx: int, op_idx: int) -> int:
        """
        Helper method to get processing time of an operation.
        
        Args:
            job_idx: Index of the job
            op_idx: Index of the operation within the job
            
        Returns:
            int: Processing time of the specified operation
        """
        return self.times_matrix[job_idx][op_idx]