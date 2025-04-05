import numpy as np
from copy import deepcopy
import time

class Operation:
    def __init__(self, machine, processing_time):
        self.machine = machine
        self.processing_time = processing_time
        self.start_time = None
        self.end_time = None

class Job:
    def __init__(self, job_id, operations):
        self.job_id = job_id
        self.operations = operations
        self.current_operation_index = 0

class JSSP:
    def __init__(self, machines_matrix, times_matrix):
        self.num_jobs = len(machines_matrix)
        self.num_machines = len(machines_matrix[0])
        self.jobs = []
        for j in range(self.num_jobs):
            operations = []
            for op in range(self.num_machines):
                operations.append(Operation(machines_matrix[j][op], times_matrix[j][op]))
            self.jobs.append(Job(j, operations))
    
    def initialize_schedule(self):
        for job in self.jobs:
            job.current_operation_index = 0
            for op in job.operations:
                op.start_time = None
                op.end_time = None

class JSSPSolverDFS_BB:
    def __init__(self, jssp_instance):
        self.jssp = jssp_instance
        self.best_schedule = None
        self.best_makespan = float('inf')
        self.nodes_explored = 0
    
    def solve(self):
        self.jssp.initialize_schedule()
        self._dfs_bb(self.jssp.jobs, [], 0)
        return self.best_schedule, self.best_makespan
    
    def _dfs_bb(self, remaining_jobs, current_schedule, current_makespan):
        self.nodes_explored += 1
        
        if not remaining_jobs:
            if current_makespan < self.best_makespan:
                self.best_makespan = current_makespan
                self.best_schedule = deepcopy(current_schedule)
            return
        
        # Critical improvement: Sort by most constrained jobs first
        remaining_jobs.sort(
            key=lambda j: (
                sum(op.processing_time for op in j.operations[j.current_operation_index:]),
                -j.current_operation_index
            ),
            reverse=True
        )
        
        for job in remaining_jobs:
            if job.current_operation_index >= len(job.operations):
                continue
            
            op = job.operations[job.current_operation_index]
            machine = op.machine
            processing_time = op.processing_time
            
            # Calculate earliest possible start time
            last_op_end = 0 if job.current_operation_index == 0 else \
                job.operations[job.current_operation_index - 1].end_time
            
            last_machine_time = 0
            for entry in current_schedule:
                if entry['machine'] == machine:
                    last_machine_time = max(last_machine_time, entry['end_time'])
            
            start_time = max(last_op_end, last_machine_time)
            end_time = start_time + processing_time
            
            # Skip if worse than current best
            if end_time >= self.best_makespan:
                continue
            
            # Create new schedule entry
            new_entry = {
                'job_id': job.job_id,
                'operation': job.current_operation_index,
                'machine': machine,
                'start_time': start_time,
                'end_time': end_time
            }
            
            # Update operation times
            op.start_time = start_time
            op.end_time = end_time
            
            # Move to next operation
            job.current_operation_index += 1
            new_remaining_jobs = [j for j in remaining_jobs if j.current_operation_index < len(j.operations)]
            
            self._dfs_bb(
                new_remaining_jobs,
                current_schedule + [new_entry],
                max(current_makespan, end_time)
            )
            
            # Backtrack
            job.current_operation_index -= 1
            op.start_time = None
            op.end_time = None
    
    def _compute_lower_bound(self, remaining_jobs, current_makespan):
        """Improved lower bound using critical path analysis"""
        # Machine workload
        machine_loads = {}
        for job in remaining_jobs:
            for op in job.operations[job.current_operation_index:]:
                machine_loads[op.machine] = machine_loads.get(op.machine, 0) + op.processing_time
        
        # Critical path
        max_job_remaining = max(
            sum(op.processing_time for op in job.operations[job.current_operation_index:])
            for job in remaining_jobs
        )
        
        return current_makespan + max(max(machine_loads.values()) if machine_loads else 0, max_job_remaining)

if __name__ == "__main__":
    
    
    machines_matrix = [
    [0, 1, 2, 3, 4],  # Job 0: M0→M1→M2→M3→M4
    [1, 2, 3, 4, 0],   # Job 1: M1→M2→M3→M4→M0
    [2, 3, 4, 0, 1],   # Job 2: M2→M3→M4→M0→M1
    [3, 4, 0, 1, 2],   # Job 3: M3→M4→M0→M1→M2
    [4, 0, 1, 2, 3]    # Job 4: M4→M0→M1→M2→M3
]

    times_matrix = [
    [5, 3, 2, 4, 1],  # Job 0: Durées (total=15)
    [3, 4, 2, 1, 5],   # Job 1: Durées (total=15)
    [2, 5, 3, 4, 1],   # Job 2: Durées (total=15)
    [4, 1, 5, 2, 3],   # Job 3: Durées (total=15)
    [1, 3, 4, 5, 2]    # Job 4: Durées (total=15)
]
    
    jssp = JSSP(machines_matrix, times_matrix)
    solver = JSSPSolverDFS_BB(jssp)
    
    start_time = time.time()
    best_schedule, best_makespan = solver.solve()
    execution_time = time.time() - start_time
    
    print(f"Optimal Makespan: {best_makespan} (Expected: 12)")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Nodes Explored: {solver.nodes_explored}")
    
    # Print machine by machine schedule
    print("\nMachine Schedules:")
    for m in range(4):
        machine_ops = [op for op in best_schedule if op['machine'] == m]
        machine_ops.sort(key=lambda x: x['start_time'])
        print(f"Machine {m}:")
        for op in machine_ops:
            print(f"  Job {op['job_id']}, Op {op['operation']}: {op['start_time']}-{op['end_time']}")