import numpy as np  
from copy import deepcopy  
import time 
from src.modules.modelisation import Operation, Job, JSSP  

class JSSPSolverDFS:
    def __init__(self, jssp_instance):
        """Initialise le solveur avec une instance du problème JSSP"""
        self.jssp = jssp_instance  
        self.best_schedule = None  
        self.best_makespan = float('inf')  
        self.nodes_explored = 0  

    def solve(self):
        """Méthode principale pour lancer la résolution"""
        self.jssp.initialize_schedule()
        self.DFS(self.jssp.jobs, [], 0)
        
        return self.best_schedule, self.best_makespan

    def lower_bound(self, remaining_jobs, current_makespan):
        """
        Borne inférieure simple : max(makespan actuel, temps total restant du job le plus long)
        """
        max_remaining_job_time = max(
            (sum(op.processing_time for op in job.operations[job.current_operation_index:])
             for job in remaining_jobs if job.current_operation_index < len(job.operations)),
            default=0
        )
        return max(current_makespan, max_remaining_job_time)

    def DFS(self, remaining_jobs, current_schedule, current_makespan):
        self.nodes_explored += 1

        # Branch and Bound : élagage si borne inférieure ≥ meilleure solution
        estimated_lb = self.lower_bound(remaining_jobs, current_makespan)
        if estimated_lb >= self.best_makespan:
            return

        # Si toutes les opérations sont planifiées
        if not remaining_jobs:
            if current_makespan < self.best_makespan:
                self.best_makespan = current_makespan
                self.best_schedule = deepcopy(current_schedule)
            return

        # Heuristique de tri : jobs avec plus de charge restante d'abord
        remaining_jobs.sort(
            key=lambda j: (
                sum(op.processing_time for op in j.operations[j.current_operation_index:]),
                -j.current_operation_index
            ),
            reverse=True
        )

        # Exploration de chaque job
        for job in remaining_jobs:
            if job.current_operation_index >= len(job.operations):
                continue

            op = job.operations[job.current_operation_index]
            machine_mapping = op.machine + 1
            processing_time = op.processing_time

            last_op_end = 0 if job.current_operation_index == 0 else \
                job.operations[job.current_operation_index - 1].end_time

            last_machine_time = max(
                (op.end_time for op in self.jssp.schedule.get(machine_mapping, [])),
                default=0
            )

            start_time = max(last_op_end, last_machine_time)
            end_time = start_time + processing_time

            # Élagage immédiat si dépasse déjà la meilleure solution
            if end_time >= self.best_makespan:
                continue

            op.start_time = start_time
            op.end_time = end_time
            self.jssp.schedule[machine_mapping].append(op)

            job.current_operation_index += 1
            new_remaining_jobs = [j for j in remaining_jobs if j.current_operation_index < len(j.operations)]

            self.DFS(
                new_remaining_jobs,
                current_schedule + [{
                    'job_id': job.job_id,
                    'operation': job.current_operation_index - 1,
                    'machine': op.machine,
                    'start_time': start_time,
                    'end_time': end_time
                }],
                max(current_makespan, end_time)
            )

            # Backtrack
            job.current_operation_index -= 1
            op.start_time = None
            op.end_time = None
            self.jssp.schedule[machine_mapping].remove(op)

if __name__ == "__main__":
    machines_matrix = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3]
    ]

    times_matrix = [
        [5, 3, 2, 4, 1],
        [3, 4, 2, 1, 5],
        [2, 5, 3, 4, 1],
        [4, 1, 5, 2, 3],
        [1, 3, 4, 5, 2]
    ]

    jssp = JSSP(machines_matrix, times_matrix)
    solver = JSSPSolverDFS(jssp)

    start_time = time.time()
    best_schedule, best_makespan = solver.solve()
    execution_time = time.time() - start_time

    # Résultats
    print(f"Optimal Makespan: {best_makespan}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Nodes Explored: {solver.nodes_explored}")

    print("\nMachine Schedules (0-indexed):")
    for m in range(len(machines_matrix[0])):
        machine_ops = [op for op in best_schedule if op['machine'] == m]
        machine_ops.sort(key=lambda x: x['start_time'])
        print(f"Machine {m}:")
        for op in machine_ops:
            print(f"  Job {op['job_id']}, Op {op['operation']}: {op['start_time']}-{op['end_time']}")
