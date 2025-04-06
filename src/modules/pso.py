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

    def generate_random_initial_sequence(self) -> List[Tuple[int, int]]:
        """Generates a valid initial sequence preserving operation order within jobs."""

        remaining_ops = deepcopy(self.jssp.job_machine_dict)

        sequence = []

        while any(remaining_ops.values()):
            available_jobs = [j for j, ops in remaining_ops.items() if ops]
            job = random.choice(available_jobs)
            op_idx = remaining_ops[job].pop(0)
            sequence.append((job, op_idx))

        return sequence
    

    def generate_spt_initial_sequence(self) -> List[Tuple[int, int]]:
        """ Generates an initial sequence using Shortest Processing Time (SPT) rule.
        Prioritizes operations with the shortest processing times while preserving job operation order.
        """
        
        # Create a list of all operations with their job, op index, and processing time
        all_ops = []
        for job_idx in self.jssp.job_machine_dict:
            for op_idx in range(len(self.jssp.job_machine_dict[job_idx])):
                processing_time = self.jssp.times_matrix[job_idx][op_idx]
                all_ops.append((job_idx, op_idx, processing_time))
        
        # Sort operations by processing time (shortest first)
        all_ops.sort(key=lambda x: x[2])
        
        # Initialize tracking of next operation needed for each job
        next_op_for_job = {job_idx: 0 for job_idx in self.jssp.job_machine_dict}
        sequence = []
        
        # Build the sequence while respecting operation order within jobs
        for job_idx, op_idx, _ in all_ops:
            # Only add if it's the next required operation for its job
            if op_idx == next_op_for_job[job_idx]:
                sequence.append((job_idx, op_idx))
                next_op_for_job[job_idx] += 1
        
        # Add any remaining operations that couldn't be added in SPT order
        # (due to operation ordering constraints)
        for job_idx in next_op_for_job:
            while next_op_for_job[job_idx] < len(self.jssp.job_machine_dict[job_idx]):
                sequence.append((job_idx, next_op_for_job[job_idx]))
                next_op_for_job[job_idx] += 1
        
        return sequence

    def optimize(
        self,
        num_particles: int = 30,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        use_spt=False
    ):
        """Run the PSO optimization."""
        particles = []
        global_best_position = None
        global_best_fitness = float("inf")

        for _ in range(num_particles):
            if use_spt and _ == 0:  # Use SPT for the first particle
                sequence = self.generate_spt_initial_sequence()
            else:
                sequence = self.generate_random_initial_sequence()
            particles.append(Particle(sequence, self.jssp.job_machine_dict))


        for iteration in range(max_iter):
            # Evaluation phase
            for particle in particles:
                particle.fitness = self.jssp.evaluate_schedule(particle.position)

                # Update personal best
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = deepcopy(particle.position)

                # Update global best
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
