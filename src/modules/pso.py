import random
import numpy as np
from copy import deepcopy
from typing import List, Tuple
import math
from src.modules.particle import Particle


DIVERSITY_THRESHOLD = 0.1  # Threshold for diversity
MIN_W = 0.2  # minimal number of iterations
MAX_MUTATION = 0.8  # Maximum mutation rate
MAX_ATTEMPTS = 100  # Maximum attempts for mutation
STAGNATION = 4  # Number of particles to reinitialize when stagnating
MAX_ATTEMPTS_MUTATION = 50


class PSOOptimizer:
    """Enhanced PSO optimization process with stagnation handling."""

    def __init__(self, jssp: "JSSP", random_seed: int = 42):
        self.jssp = jssp
        self.iteration_history = []
        self.makespan_history = []
        self.diversity_history = []
        self.stagnation_count = 0
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def generate_initial_sequence(self, cluster_size: int = 3) -> List[Tuple[int, int]]:
        """Cluster approach that balances machine workload during clustering."""
        remaining_ops = deepcopy(self.jssp.job_machine_dict)
        sequence = []
        machine_counts = {m: 0 for m in range(1, self.jssp.num_machines + 1)}

        while any(ops_left for ops_left in remaining_ops.values()):
            available_ops = []
            for job_idx, ops_left in remaining_ops.items():
                if ops_left:
                    op_idx = ops_left[0]
                    op = self.jssp.jobs[job_idx].operations[op_idx]
                    available_ops.append(
                        (job_idx, op_idx, op.processing_time, op.machine)
                    )

            # Balance clusters by machine distribution
            clusters = []
            current_cluster = []
            machine_in_cluster = set()

            for op in sorted(available_ops, key=lambda x: machine_counts[x[3]]):
                if (
                    len(current_cluster) < cluster_size
                    and op[3] not in machine_in_cluster
                ):
                    current_cluster.append(op)
                    machine_in_cluster.add(op[3])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [op]
                    machine_in_cluster = {op[3]}
            if current_cluster:
                clusters.append(current_cluster)

            # Process clusters
            for cluster in clusters:
                # Sort by both processing time and machine load
                cluster_sorted = sorted(
                    cluster, key=lambda x: (machine_counts[x[3]], x[2])
                )

                for op in cluster_sorted:
                    job_idx, op_idx, _, machine = op
                    if remaining_ops[job_idx] and op_idx == remaining_ops[job_idx][0]:
                        sequence.append((job_idx, op_idx))
                        remaining_ops[job_idx].pop(0)
                        machine_counts[machine] += 1

        return sequence

    def calculate_diversity(self, particles: List["Particle"]) -> float:
        """Calculate population diversity based on position differences."""
        if not particles:
            return 0.0

        centroid = [0] * len(particles[0].position)  # Initialisation du centroid

        # Calcul du centroïde
        for particle in particles:
            for i, (job, machine) in enumerate(particle.position):
                centroid[
                    i
                ] += job  # chaque élément du centroid est un total des ID des jobs pour toutes les particules.

        centroid = [x / len(particles) for x in centroid]  # Moyenne des IDs de jobs.

        diversity = 0.0
        for particle in particles:
            distance = sum(
                (p[0] - c) ** 2 for p, c in zip(particle.position, centroid)
            )  # mesures la distance euclidienne entre chaque particule et le centroid
            diversity += math.sqrt(distance)

        # diversité moyenne
        return diversity / len(particles)

    def is_sequence_valid(self, sequence: List[Tuple[int, int]], job_id: int) -> bool:
        """Check if operations for a job are in correct machine order."""
        job_ops = [op[1] for op in sequence if op[0] == job_id]
        return job_ops == self.jssp.job_machine_dict[job_id]

    def handle_stagnation(
        self, particles: List["Particle"], global_best_position: List[Tuple[int, int]]
    ):
        """Diversification strategies when stagnating."""
        # Reinitialize worst particles
        particles.sort(key=lambda p: p.best_fitness)
        num_to_reinit = max(
            1, len(particles) // STAGNATION
        )  # On choisit un nombre de particules à réinitialiser basé sur le taux de stagnation.

        for i in range(-num_to_reinit, 0):
            # Set seed for reinitialized particles
            particle_seed = self.random_seed + len(particles) + i
            random.seed(particle_seed)
            particles[i] = Particle(
                self.generate_initial_sequence(),
                self.jssp.job_machine_dict,
                random_seed=i,
            )

        # Add perturbed global best
        perturbed = self.perturb_solution(
            global_best_position, max(len(global_best_position) // 5, 1)
        )
        particles[-1] = Particle(perturbed, self.jssp.job_machine_dict, random_seed=80)
        random.seed(self.random_seed)  # Reset to main seed

    def perturb_solution(
        self, solution: List[Tuple[int, int]], num_swaps: int = 3
    ) -> List[Tuple[int, int]]:
        """Create a slightly modified version of the best solution."""
        perturbed = deepcopy(solution)
        swaps_applied = 0
        attempts = 0

        while swaps_applied < num_swaps and attempts < MAX_ATTEMPTS:
            i, j = random.sample(
                range(len(perturbed)), 2
            )  # choisit deux indices distincts i et j dans la solution perturbée.
            if (
                perturbed[i][0] != perturbed[j][0]
            ):  # Assure que les jobs sont différents
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

                if self.is_sequence_valid(
                    perturbed,
                    perturbed[i][0],
                ) and self.is_sequence_valid(perturbed, perturbed[j][0]):
                    swaps_applied += 1
                else:
                    # Undo invalid swap
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            attempts += 1

        return perturbed

    def optimize(
        self,
        num_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        adaptive_params: bool = True,
        mutation_rate: float = 0.1,
        max_stagnation: int = 15,
        early_stopping_window: int = 20,
        improvement_threshold: float = 0.01,
    ):
        """Run the enhanced PSO optimization."""
        particles = []
        global_best_position = None
        global_best_fitness = float("inf")
        self.stagnation_count = 0

        # Initialize swarm with varied seeds for each particle
        for i in range(num_particles):
            # Set a unique seed for each particle based on the main seed + particle index
            particle_seed = self.random_seed + i
            random.seed(particle_seed)

            sequence = self.generate_initial_sequence()
            particles.append(
                Particle(
                    sequence, self.jssp.job_machine_dict, random_seed=particle_seed
                )
            )

        # Reset to main seed for the rest of the optimization
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        for iteration in range(max_iter):
            # Adaptive parameters
            current_w = w
            current_mutation = mutation_rate
            if adaptive_params:
                current_w = w * (1 - iteration / max_iter) + MIN_W * (
                    iteration / max_iter
                )
                current_mutation = min(
                    MAX_MUTATION,
                    mutation_rate * (1 + self.stagnation_count / 10),
                )

            # Evaluate particles
            improved = False
            for particle in particles:
                particle.fitness = self.jssp.evaluate_schedule(particle.position)

                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = deepcopy(particle.position)

                if particle.fitness < global_best_fitness:
                    global_best_fitness = particle.fitness
                    global_best_position = deepcopy(particle.position)
                    improved = True

            # Update stagnation counter
            self.stagnation_count = 0 if improved else self.stagnation_count + 1

            # Record metrics
            self.iteration_history.append(iteration)
            self.makespan_history.append(global_best_fitness)
            self.diversity_history.append(self.calculate_diversity(particles))

            # Update particles
            for particle in particles:
                particle.update_velocity(
                    global_best_position,
                    current_w,
                    c1,
                    c2,
                    mutation_rate,
                )
                particle.update_position()
                particle.apply_mutation(current_mutation)

            # Stagnation handling
            if self.stagnation_count >= max_stagnation:
                self.handle_stagnation(particles, global_best_position)
                self.stagnation_count = 0

            # Early stopping
            if (
                early_stopping_window
                and len(self.makespan_history) >= early_stopping_window
            ):
                window_min = min(self.makespan_history[-early_stopping_window:])
                if (
                    global_best_fitness - window_min
                ) < improvement_threshold * global_best_fitness:
                    print(f"Early stopping at iteration {iteration}")
                    break

            # Progress reporting
            if iteration % 10 == 0 or iteration == max_iter - 1:
                print(
                    f"Iter {iteration}: Best={global_best_fitness:.1f} "
                    f"Div={self.diversity_history[-1]:.2f} "
                    f"Stag={self.stagnation_count}/{max_stagnation}"
                )

        return global_best_position, global_best_fitness
