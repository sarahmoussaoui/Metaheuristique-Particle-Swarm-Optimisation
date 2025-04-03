from modelisation import JSSP
from modelisation import machines_matrix, times_matrix

class JSSPSolver:
    """Résout le problème JSSP en utilisant DFS."""
    def __init__(self, jssp: JSSP): 
        self.jssp = jssp
        self.best_makespan = float('inf')
        self.best_schedule = None

        # Créer un mapping des machines avec des indices valides
        self.machine_mapping = self.create_machine_mapping()

    def create_machine_mapping(self):
        """ Crée un mapping entre les identifiants des machines et des indices. """
        machine_set = set()
        for job in self.jssp.jobs:
            for operation in job.operations:
                machine_set.add(operation.machine)
        # Créer un dictionnaire associant chaque identifiant de machine à un indice
        machine_mapping = {machine: idx for idx, machine in enumerate(sorted(machine_set))}
        print(f"Mapping des machines : {machine_mapping}")
        return machine_mapping

    def dfs(self, schedule, machine_times, job_times):
        """ DFS récursif pour explorer toutes les séquences possibles. """
        if len(schedule) == sum(len(job.operations) for job in self.jssp.jobs):
            makespan = max(machine_times)
            if makespan < self.best_makespan:
                self.best_makespan = makespan
                self.best_schedule = schedule[:]
            return

        for job in self.jssp.jobs:
            task_idx = job.current_operation_index
            if task_idx < len(job.operations):
                operation = job.operations[task_idx]
                machine = operation.machine
                duration = operation.processing_time

                # Utiliser le mapping pour obtenir l'indice de la machine
                machine_idx = self.machine_mapping[machine]

                # Vérifier si l'index de la machine est valide
                if machine_idx >= len(machine_times):
                    print(f"Index de machine invalide : {machine_idx}")
                    continue

                # Calculer les temps de début et de fin
                start_time = max(machine_times[machine_idx], job_times[job.job_id])
                end_time = start_time + duration

                machine_times[machine_idx] = end_time
                job_times[job.job_id] = end_time
                job.current_operation_index += 1

                # Appel récursif avec la tâche ajoutée
                self.dfs(schedule + [(job.job_id, task_idx, machine_idx, start_time)], machine_times, job_times)

                # Revenir en arrière (backtracking)
                machine_times[machine_idx] -= duration
                job_times[job.job_id] -= duration
                job.current_operation_index -= 1

    def solve(self):
        """ Lance la recherche DFS. """
        initial_machine_times = [0] * len(self.machine_mapping)  # Ajuster la taille pour le mapping
        initial_job_times = [0] * self.jssp.num_jobs
        self.dfs([], initial_machine_times, initial_job_times)
        return self.best_schedule, self.best_makespan
    
jssp = JSSP(machines_matrix, times_matrix)

# Résolution avec DFS
solver = JSSPSolver(jssp)
schedule, makespan = solver.solve()

# Affichage des résultats
print("Meilleur ordonnancement :", schedule)
print("Makespan optimal :", makespan)