from copy import deepcopy
from typing import List, Tuple
import random


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
