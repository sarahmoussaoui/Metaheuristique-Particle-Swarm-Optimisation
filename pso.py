import numpy as np
import random
from copy import deepcopy
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from modelisation import Operation, Job, JSSP
from visualisation import ScheduleVisualizer


class Particle:
    """Represents a particle in the PSO algorithm."""

    def __init__(self, sequence: List[Tuple[int, int]]):
        self.position = sequence
        self.velocity = []
        self.best_position = deepcopy(sequence)
        self.best_fitness = float("inf")
        self.fitness = float("inf")

    def update_position(self):
        """Update position by applying velocity (swaps)."""
        new_position = deepcopy(self.position)
        for i, j in self.velocity:
            if i < len(new_position) and j < len(new_position):
                new_position[i], new_position[j] = new_position[j], new_position[i]
        self.position = new_position
        if len(self.velocity) > len(self.position):
            self.velocity = self.velocity[: len(self.position)]

    def update_velocity(
        self,
        global_best_position: List[Tuple[int, int]],
        w: float,
        c1: float,
        c2: float,
    ):
        """Update velocity based on personal and global best positions."""
        new_velocity = []
        for swap in self.velocity:
            if random.random() < w:
                new_velocity.append(swap)

        for i in range(len(self.position)):
            if random.random() < c1 and i < len(self.best_position):
                if self.position[i] != self.best_position[i]:
                    j = self.best_position.index(self.position[i])
                    new_velocity.append((i, j))

        for i in range(len(self.position)):
            if random.random() < c2 and i < len(global_best_position):
                if self.position[i] != global_best_position[i]:
                    j = global_best_position.index(self.position[i])
                    new_velocity.append((i, j))

        self.velocity = new_velocity


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


# Example usage
if __name__ == "__main__":
    machines_matrix = [
        [3, 2, 4, 1],  # Job 0 operations
        [1, 4, 2, 3],  # Job 1 operations
        [2, 1, 3, 4],  # Job 2 operations
    ]

    times_matrix = [
        [5, 10, 8, 12],  # Job 0 processing times
        [7, 9, 6, 14],  # Job 1 processing times
        [4, 11, 5, 8],  # Job 2 processing times
    ]

    # Create and run optimization
    jssp = JSSP(machines_matrix, times_matrix)
    optimizer = PSOOptimizer(jssp)
    best_schedule, best_makespan = optimizer.optimize(num_particles=20, max_iter=50)

    # Display results
    print("\nBest Schedule Found:")
    print(best_schedule)
    print(f"Best Makespan: {best_makespan}")

    # Visualize results
    jssp.evaluate_schedule(best_schedule)
    ScheduleVisualizer.plot_convergence(
        optimizer.iteration_history, optimizer.makespan_history
    )
    ScheduleVisualizer.plot_gantt_chart(jssp)
