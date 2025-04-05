from copy import deepcopy
from typing import List, Tuple
import random


class Particle:
    """Represents a particle in the PSO algorithm."""

    def __init__(
        self, sequence: List[Tuple[int, int]], job_machine_dict: dict[int, list[int]]
    ):
        self.position = sequence
        self.velocity = []
        self.best_position = deepcopy(sequence)
        self.best_fitness = float("inf")
        self.fitness = float("inf")

    from copy import deepcopy

    def update_position(self, job_machine_dict: dict[int, list[int]]):
        """Update position by applying velocity (swaps), skipping invalid ones and preserving machine order."""
        new_position = deepcopy(self.position)
        valid_velocity = []

        for i, j in self.velocity:
            if (
                i < len(new_position)
                and j < len(new_position)
                and new_position[i][0] != new_position[j][0]
            ):
                # Tentatively swap
                new_position[i], new_position[j] = new_position[j], new_position[i]

                def is_machine_order_valid(pos, job_id):
                    machines_in_position = [op[1] for op in pos if op[0] == job_id]
                    correct_order = job_machine_dict[job_id]
                    return all(
                        m1 == m2 for m1, m2 in zip(machines_in_position, correct_order)
                    )

                job_i = new_position[i][0]
                job_j = new_position[j][0]

                if is_machine_order_valid(
                    new_position, job_i
                ) and is_machine_order_valid(new_position, job_j):
                    valid_velocity.append((i, j))  # Swap is valid, keep it
                else:
                    # Revert the swap if order is broken
                    new_position[i], new_position[j] = new_position[j], new_position[i]

        self.position = new_position
        self.velocity = valid_velocity[: len(new_position)]  # Trim if needed

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
            if random.random() < random.random() * c1:
                if self.position[i] != self.best_position[i]:
                    j = self.best_position.index(self.position[i])
                    new_velocity.append((i, j))

        for i in range(len(self.position)):
            if random.random() < random.random() * c2:
                if self.position[i] != global_best_position[i]:
                    j = global_best_position.index(self.position[i])
                    new_velocity.append((i, j))

        self.velocity = new_velocity
