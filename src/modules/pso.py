import random
from copy import deepcopy
from typing import List, Tuple
from src.modules.modelisation import JSSP
from src.modules.particle import Particle
import math


class PSOOptimizer:
    """Enhanced PSO optimization process with stagnation handling."""

    def __init__(self, jssp: JSSP):
        self.jssp = jssp
        self.iteration_history = []
        self.makespan_history = []
        self.diversity_history = []
        self.stagnation_count = 0

    def generate_initial_sequence(self) -> List[Tuple[int, int]]:
        """Generates a valid initial sequence preserving operation order within jobs."""
        remaining_ops = deepcopy(self.jssp.job_machine_dict)
        sequence = []

        while any(remaining_ops.values()):
            available_jobs = [j for j, ops in remaining_ops.items() if ops]
            job = random.choice(available_jobs)
            op_idx = remaining_ops[job].pop(0)
            sequence.append((job, op_idx))

        return sequence

    def calculate_diversity(self, particles: List[Particle]) -> float:
        """Calculate population diversity based on position differences."""
        if not particles:
            return 0.0

        centroid = [0] * len(particles[0].position)
        for particle in particles:
            for i, (job, machine) in enumerate(particle.position):
                centroid[i] += job

        centroid = [x / len(particles) for x in centroid]

        diversity = 0.0
        for particle in particles:
            distance = sum((p[0] - c) ** 2 for p, c in zip(particle.position, centroid))
            diversity += math.sqrt(distance)

        return diversity / len(particles)

    def apply_mutation(self, particle: Particle, mutation_rate: float = 0.1):
        """Apply mutation to a particle to maintain diversity."""
        if random.random() < mutation_rate:
            attempts = 0
            while attempts < 10:
                i, j = random.sample(range(len(particle.position)), 2)
                if particle.position[i][0] != particle.position[j][0]:
                    new_position = particle.position.copy()
                    new_position[i], new_position[j] = new_position[j], new_position[i]

                    if self.is_sequence_valid(
                        new_position, new_position[i][0]
                    ) and self.is_sequence_valid(new_position, new_position[j][0]):
                        particle.position = new_position
                        break
                attempts += 1

    def is_sequence_valid(self, sequence: List[Tuple[int, int]], job_id: int) -> bool:
        """Check if operations for a job are in correct machine order."""
        job_ops = [op[1] for op in sequence if op[0] == job_id]
        return job_ops == self.jssp.job_machine_dict[job_id]

    def handle_stagnation(
        self, particles: List[Particle], global_best_position: List[Tuple[int, int]]
    ):
        """Diversification strategies when stagnating."""
        # Reinitialize worst particles
        particles.sort(key=lambda p: p.best_fitness)
        num_to_reinit = max(1, len(particles) // 5)

        for i in range(-num_to_reinit, 0):
            particles[i] = Particle(
                self.generate_initial_sequence(), self.jssp.job_machine_dict
            )

        # Add perturbed global best
        perturbed = self.perturb_solution(global_best_position)
        particles[-1] = Particle(perturbed, self.jssp.job_machine_dict)

    def perturb_solution(
        self, solution: List[Tuple[int, int]], num_swaps: int = 3
    ) -> List[Tuple[int, int]]:
        """Create a slightly modified version of the best solution."""
        perturbed = deepcopy(solution)
        swaps_applied = 0
        attempts = 0

        while swaps_applied < num_swaps and attempts < 20:
            i, j = random.sample(range(len(perturbed)), 2)
            if perturbed[i][0] != perturbed[j][0]:
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

                if self.is_sequence_valid(
                    perturbed, perturbed[i][0]
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

        # Initialize swarm
        for _ in range(num_particles):
            sequence = self.generate_initial_sequence()
            particles.append(Particle(sequence, self.jssp.job_machine_dict))

        for iteration in range(max_iter):
            # Adaptive parameters
            current_w = w
            current_mutation = mutation_rate
            if adaptive_params:
                current_w = w * (1 - iteration / max_iter) + 0.4 * (
                    iteration / max_iter
                )
                current_mutation = min(
                    0.5, mutation_rate * (1 + self.stagnation_count / 10)
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
                particle.update_velocity(global_best_position, current_w, c1, c2)
                particle.update_position()
                self.apply_mutation(particle, current_mutation)

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
