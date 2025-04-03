import random
from copy import deepcopy
from typing import List, Tuple
from src.modules.modelisation import JSSP
from src.modules.particle import Particle


class PSOOptimizer:
    """Handles the PSO optimization process."""

    def __init__(self, jssp: JSSP):
        self.jssp = jssp
        self.iteration_history = []
        self.makespan_history = []

    @staticmethod
    def generate_initial_sequence(jssp: JSSP) -> List[Tuple[int, int]]:
        """Generate a random but valid initial sequence of operations."""
        sequence = []
        for job_idx in range(jssp.num_jobs):
            for op_idx in range(len(jssp.jobs[job_idx].operations)):
                sequence.append((job_idx, op_idx))
        random.shuffle(sequence)
        return sequence

    def optimize(
        self,
        num_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        """Run the PSO optimization."""
        particles = []
        global_best_position = None
        global_best_fitness = float("inf")

        for _ in range(num_particles):
            sequence = self.generate_initial_sequence(self.jssp)
            particles.append(Particle(sequence))

        for iteration in range(max_iter):
            for particle in particles:
                particle.fitness = self.jssp.evaluate_schedule(particle.position)

                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = deepcopy(particle.position)

                if particle.fitness < global_best_fitness:
                    global_best_fitness = particle.fitness
                    global_best_position = deepcopy(particle.position)

            self.iteration_history.append(iteration)
            self.makespan_history.append(global_best_fitness)

            for particle in particles:
                particle.update_velocity(global_best_position, w, c1, c2)
                particle.update_position()

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Makespan: {global_best_fitness}")

        return global_best_position, global_best_fitness
