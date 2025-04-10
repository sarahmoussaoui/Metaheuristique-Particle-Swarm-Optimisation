from copy import deepcopy
from typing import List, Tuple, Dict
import random
import numpy as np
import math
from collections import defaultdict

MAX_ATTEMPTS_MUTATION = 50


class Particle:
    """Enhanced Particle with improved exploration capabilities."""

    def __init__(
        self,
        sequence: List[Tuple[int, int]],
        job_machine_dict: Dict[int, List[int]],
        max_velocity_size: int = None,
        random_seed: int = None,
    ):
        self.position = sequence
        self.velocity = []
        self.best_position = deepcopy(sequence)
        self.best_fitness = float("inf")
        self.fitness = float("inf")
        self.job_machine_dict = job_machine_dict
        self.max_velocity_size = max_velocity_size or max(5, len(sequence) // 2)
        self.rng = random.Random(random_seed) if random_seed else random.Random()

        self.initialize_velocity()

    def initialize_velocity(self):
        """Initialize velocity with diverse valid swaps."""
        self.velocity = []
        valid_swaps = []

        # Generate all possible valid swaps
        for i in range(len(self.position)):
            for j in range(i + 1, len(self.position)):
                if self.position[i][0] != self.position[j][0]:
                    valid_swaps.append((i, j))

        # Sample without replacement if enough swaps exist
        if valid_swaps:
            num_swaps = min(self.max_velocity_size, len(valid_swaps))
            self.velocity = self.rng.sample(valid_swaps, num_swaps)

    def is_machine_order_valid(
        self, position: List[Tuple[int, int]], job_id: int
    ) -> bool:
        """Check if machine order for a job is valid."""
        machines_in_position = [op[1] for op in position if op[0] == job_id]
        correct_order = self.job_machine_dict[job_id]
        return machines_in_position == correct_order

    def update_position(self) -> int:
        """Update position by applying velocity with repair mechanism."""
        new_position = deepcopy(self.position)
        applied_swaps = 0

        for i, j in self.velocity:
            new_position[i], new_position[j] = new_position[j], new_position[i]

            # Check constraints
            valid_i = self.is_machine_order_valid(new_position, new_position[i][0])
            valid_j = self.is_machine_order_valid(new_position, new_position[j][0])

            if not (valid_i and valid_j):
                # Undo invalid swap
                new_position[i], new_position[j] = new_position[j], new_position[i]
            else:
                applied_swaps += 1

        self.position = new_position
        return applied_swaps

    def update_velocity(
        self,
        global_best_position: List[Tuple[int, int]],
        w: float,
        c1: float,
        c2: float,
        mutation_rate: float = 0.1,
    ):
        """Update velocity with enhanced exploration."""
        new_velocity = []
        used_indices = set()

        # Helper to check if swap can be added
        def can_add_swap(i, j):
            return (
                i != j
                and i not in used_indices
                and j not in used_indices
                and self.position[i][0] != self.position[j][0]
            )

        # 1. Inertia component (keep some existing swaps)
        for swap in self.velocity:
            if self.rng.random() < w and len(new_velocity) < self.max_velocity_size:
                i, j = swap
                if can_add_swap(i, j):
                    new_velocity.append(swap)
                    used_indices.update([i, j])

        # 2. Cognitive component (personal best)
        diff_indices = [
            i
            for i in range(len(self.position))
            if self.position[i] != self.best_position[i]
        ]

        if diff_indices and self.rng.random() < c1:
            for i in self.rng.sample(diff_indices, min(3, len(diff_indices))):
                try:
                    j = self.best_position.index(self.position[i])
                    if (
                        can_add_swap(i, j)
                        and len(new_velocity) < self.max_velocity_size
                    ):
                        new_velocity.append((i, j))
                        used_indices.update([i, j])
                except ValueError:
                    pass

        # 3. Social component (global best)
        diff_indices = [
            i
            for i in range(len(self.position))
            if self.position[i] != global_best_position[i]
        ]

        if diff_indices and self.rng.random() < c2:
            for i in self.rng.sample(diff_indices, min(3, len(diff_indices))):
                try:
                    j = global_best_position.index(self.position[i])
                    if (
                        can_add_swap(i, j)
                        and len(new_velocity) < self.max_velocity_size
                    ):
                        new_velocity.append((i, j))
                        used_indices.update([i, j])
                except ValueError:
                    pass

        # 4. Mutation component
        if (
            self.rng.random() < mutation_rate
            and len(new_velocity) < self.max_velocity_size
        ):
            attempts = 0
            while (
                attempts < MAX_ATTEMPTS_MUTATION
                and len(new_velocity) < self.max_velocity_size
            ):
                i, j = self.rng.sample(range(len(self.position)), 2)
                if (
                    can_add_swap(i, j)
                    and (i, j) not in new_velocity
                    and (j, i) not in new_velocity
                ):
                    new_velocity.append((i, j))
                    used_indices.update([i, j])
                    break
                attempts += 1

        self.velocity = new_velocity[: self.max_velocity_size]

    def apply_mutation(self, mutation_rate=0.1):
        """Enhanced mutation with more diverse operations."""
        if self.rng.random() < mutation_rate:
            # 50% chance for block mutation
            if self.rng.random() < 0.5:
                block_size = self.rng.randint(2, min(10, len(self.position) // 3))
                start = self.rng.randint(0, len(self.position) - block_size)
                end = start + block_size

                # Extract block and group by job
                block = self.position[start:end]
                job_ops = defaultdict(list)
                for op in block:
                    job_ops[op[0]].append(op[1])

                # Validate block can be shuffled
                valid = True
                for job, ops in job_ops.items():
                    if ops != self.job_machine_dict[job][: len(ops)]:
                        valid = False
                        break

                if valid:
                    # Shuffle jobs while maintaining internal order
                    jobs = list(job_ops.keys())
                    self.rng.shuffle(jobs)

                    # Rebuild block
                    new_block = []
                    for job in jobs:
                        new_block.extend((job, op) for op in job_ops[job])

                    self.position[start:end] = new_block
            else:
                # Multiple swaps mutation
                num_swaps = self.rng.randint(1, 3)
                for _ in range(num_swaps):
                    attempts = 0
                    while attempts < MAX_ATTEMPTS_MUTATION:
                        i, j = self.rng.sample(range(len(self.position)), 2)
                        if self.position[i][0] != self.position[j][0]:
                            new_position = self.position.copy()
                            new_position[i], new_position[j] = (
                                new_position[j],
                                new_position[i],
                            )
                            if self.is_machine_order_valid(
                                new_position, new_position[i][0]
                            ) and self.is_machine_order_valid(
                                new_position, new_position[j][0]
                            ):
                                self.position = new_position
                                break
                        attempts += 1
