from copy import deepcopy
from typing import List, Tuple, Dict
import random
import numpy as np
import math
from src.modules.modelisation import JSSP
from src.modules.particle import Particle

MAX_ATTEMPTS_MUTATION = 50
DIVERSITY_THRESHOLD = 0.2
MIN_W = 0.3
MAX_MUTATION = 0.75
STAGNATION_THRESHOLD = 15
EARLY_STOPPING_WINDOW = 25
IMPROVEMENT_THRESHOLD = 0.005
C_MAX = 1.7  # Changed to be the max value for c1 and c2
C_MIN = 0.2
W_MAX = 0.8
W_MIN = 0.3


class PSOOptimizer:
    """Enhanced PSO with reproducible results and improved convergence."""

    def __init__(self, jssp: JSSP, random_seed: int = 42):
        self.jssp = jssp
        self.iteration_history = []
        self.makespan_history = []
        self.diversity_history = []
        self.stagnation_count = 0
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        np.random.seed(random_seed)
        self.max_possible_diversity = None

    def calculate_max_possible_diversity(self, num_particles: int) -> float:
        """Calculate theoretical maximum diversity for deterministic adaptation."""
        num_jobs = len(self.jssp.job_machine_dict)
        num_ops = sum(len(ops) for ops in self.jssp.job_machine_dict.values())

        if num_particles < 2:
            return 0.0
        return math.sqrt(num_jobs * num_ops) * math.log(num_particles + 1)

    def generate_initial_sequence(self, cluster_size: int = 3) -> List[Tuple[int, int]]:
        """Generate initial sequence with machine load balancing."""
        remaining_ops = {
            job: ops.copy() for job, ops in self.jssp.job_machine_dict.items()
        }
        sequence = []
        machine_loads = {m: 0 for m in range(1, self.jssp.num_machines + 1)}

        while any(ops_left for ops_left in remaining_ops.values()):
            available = []
            for job, ops_left in remaining_ops.items():
                if ops_left:
                    op_idx = ops_left[0]
                    machine = self.jssp.jobs[job].operations[op_idx].machine
                    pt = self.jssp.jobs[job].operations[op_idx].processing_time
                    available.append((job, op_idx, machine, pt))

            clusters = []
            current_cluster = []
            current_machines = set()

            for op in sorted(available, key=lambda x: machine_loads[x[2]]):
                if (
                    len(current_cluster) < cluster_size
                    and op[2] not in current_machines
                ):
                    current_cluster.append(op)
                    current_machines.add(op[2])
                else:
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = [op]
                    current_machines = {op[2]}

            if current_cluster:
                clusters.append(current_cluster)

            for cluster in clusters:
                cluster_sorted = sorted(
                    cluster, key=lambda x: (machine_loads[x[2]], x[3])
                )

                for op in cluster_sorted:
                    job, op_idx, machine, pt = op
                    if remaining_ops[job] and op_idx == remaining_ops[job][0]:
                        sequence.append((job, op_idx))
                        remaining_ops[job].pop(0)
                        machine_loads[machine] += pt

        return sequence

    def calculate_diversity(self, particles: List[Particle]) -> float:
        """Calculate population diversity using position differences."""
        if len(particles) < 2:
            return 0.0

        positions = np.zeros((len(particles), len(particles[0].position), 2))
        for i, p in enumerate(particles):
            for j, (job, op) in enumerate(p.position):
                positions[i, j, 0] = job
                positions[i, j, 1] = op

        centroid = np.mean(positions, axis=0)
        distances = []
        for p in positions:
            diff = p - centroid
            distance = np.sqrt(np.sum(diff**2))
            distances.append(distance)

        return np.mean(distances)

    def is_sequence_valid(self, sequence: List[Tuple[int, int]], job_id: int) -> bool:
        """Check if operations for a job are in correct order."""
        job_ops = [op[1] for op in sequence if op[0] == job_id]
        return job_ops == self.jssp.job_machine_dict[job_id]

    def handle_stagnation(
        self, particles: List[Particle], global_best_position: List[Tuple[int, int]]
    ):
        """Diversification strategy for stagnation handling."""
        particles.sort(key=lambda p: p.best_fitness)
        num_to_replace = max(1, len(particles) // 4)

        for i in range(-num_to_replace, 0):
            if self.rng.random() < 0.7:
                new_seed = self.random_seed + len(particles) + i + 1
                particles[i] = Particle(
                    self.generate_initial_sequence(),
                    self.jssp.job_machine_dict,
                    random_seed=new_seed,
                )
            else:
                perturbed = self.perturb_solution(
                    global_best_position,
                    num_swaps=max(len(global_best_position) // 4, 3),
                )
                new_seed = self.random_seed + len(particles) * 2 + i + 1
                particles[i] = Particle(
                    perturbed, self.jssp.job_machine_dict, random_seed=new_seed
                )

    def perturb_solution(
        self, solution: List[Tuple[int, int]], num_swaps: int = 3
    ) -> List[Tuple[int, int]]:
        """Create a modified version of the solution."""
        perturbed = deepcopy(solution)
        swaps_done = 0
        attempts = 0

        while swaps_done < num_swaps and attempts < MAX_ATTEMPTS_MUTATION:
            i, j = self.rng.sample(range(len(perturbed)), 2)
            if perturbed[i][0] != perturbed[j][0]:
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

                if self.is_sequence_valid(
                    perturbed, perturbed[i][0]
                ) and self.is_sequence_valid(perturbed, perturbed[j][0]):
                    swaps_done += 1
                else:
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            attempts += 1

        return perturbed

    def optimize(
        self,
        num_particles: int = 30,
        max_iter: int = 200,
        w: Tuple[float, float] = (W_MIN, W_MAX),  # Changed to tuple
        c1: Tuple[float, float] = (C_MIN, C_MAX),  # Changed to tuple
        c2: Tuple[float, float] = (C_MIN, C_MAX),  # Changed to tuple
        adaptive_params: bool = True,
        mutation_rate: float = 0.15,
        max_stagnation: int = STAGNATION_THRESHOLD,
        early_stopping_window: int = EARLY_STOPPING_WINDOW,
        improvement_threshold: float = IMPROVEMENT_THRESHOLD,
    ):
        """Run the optimization with reproducible adaptive parameters."""
        particles = []
        global_best_position = None
        global_best_fitness = float("inf")
        self.stagnation_count = 0
        self.max_possible_diversity = self.calculate_max_possible_diversity(
            num_particles
        )

        # Initialize swarm with unique seeds
        for i in range(num_particles):
            particle_seed = self.random_seed + i * 997  # Large prime for better spread
            self.rng.seed(particle_seed)

            sequence = self.generate_initial_sequence()
            particles.append(
                Particle(
                    sequence, self.jssp.job_machine_dict, random_seed=particle_seed
                )
            )

        self.rng.seed(self.random_seed)  # Reset main RNG

        for iteration in range(max_iter):
            # Adaptive parameters
            current_w = w[1]  # Default to max value
            current_mutation = mutation_rate
            current_c1 = c1[1]  # Default to max value
            current_c2 = c2[1]  # Default to max value

            if adaptive_params:
                progress = iteration / max_iter

                # Calculate current parameters based on progress
                current_w = w[1] - (w[1] - w[0]) * progress**2
                current_c1 = c1[1] - (c1[1] - c1[0]) * progress**1.5
                current_c2 = c2[0] + (c2[1] - c2[0]) * progress**0.7

                # Reproducible mutation adaptation
                if self.max_possible_diversity > 0 and self.diversity_history:
                    diversity_ratio = min(
                        1.0, self.diversity_history[-1] / self.max_possible_diversity
                    )
                    current_mutation = min(
                        MAX_MUTATION, mutation_rate * (1 + (1 - diversity_ratio) * 4)
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
                    current_c1,
                    current_c2,
                    current_mutation,
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
                    f"Iter {iteration:3d}: Best={global_best_fitness:7.1f} "
                    f"Div={self.diversity_history[-1]:5.2f} "
                    f"w={current_w:.2f} mut={current_mutation:.2f} c1={current_c1:.2f} c2={current_c2:.2f} "
                    f"Stag={self.stagnation_count:2d}/{max_stagnation}"
                )

        return global_best_position, global_best_fitness
