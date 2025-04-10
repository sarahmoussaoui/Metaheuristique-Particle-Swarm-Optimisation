from copy import deepcopy
from typing import List, Tuple
import random


MAX_ATTEMPTS_MUTATION = 50


class Particle:
    """Represents a particle in the PSO algorithm with improved diversity mechanisms."""

    def __init__(
        self,
        sequence: List[Tuple[int, int]],
        job_machine_dict: dict[int, list[int]],
        max_velocity_size: int = None,
        random_seed: int = None,
    ):
        self.position = sequence
        self.velocity = []
        self.best_position = deepcopy(sequence)
        self.best_fitness = float("inf")
        self.fitness = float("inf")
        self.job_machine_dict = job_machine_dict
        self.max_velocity_size = max_velocity_size or max(1, len(sequence) * 2)
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)
        self.initialize_velocity()

    def initialize_velocity(self):
        """Initialize velocity with random swaps that respect constraints."""
        self.velocity = []
        attempts = 0
        max_attempts = 100

        while len(self.velocity) < self.max_velocity_size and attempts < max_attempts:
            attempts += 1
            i, j = random.sample(range(len(self.position)), 2)

            # Skip if same job or invalid indices
            if self.position[i][0] == self.position[j][0]:
                continue

            # Check if swap would maintain valid machine order
            temp_position = self.position.copy()
            temp_position[i], temp_position[j] = temp_position[j], temp_position[i]

            if self.is_machine_order_valid(
                temp_position, temp_position[i][0]
            ) and self.is_machine_order_valid(temp_position, temp_position[j][0]):
                self.velocity.append((i, j))

    def is_machine_order_valid(
        self, position: List[Tuple[int, int]], job_id: int
    ) -> bool:
        """Check if machine order for a job is valid."""
        machines_in_position = [op[1] for op in position if op[0] == job_id]
        correct_order = self.job_machine_dict[job_id]
        return machines_in_position == correct_order

    def update_position(self):
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
        """Update velocity with enhanced diversity mechanisms."""
        new_velocity = []
        used_indices = set()

        # Helper function to check if swap can be added
        def can_add_swap(i, j):
            return (
                i != j
                and i not in used_indices
                and j not in used_indices
                and self.position[i][0] != self.position[j][0]
            )

        # 1. Inertia component (keep some existing swaps)
        r = random.random()
        r1 = random.random()
        r2 = random.random()
        for swap in self.velocity:
            if r < w and len(new_velocity) < self.max_velocity_size:
                i, j = swap
                if can_add_swap(i, j):
                    new_velocity.append(swap)
                    used_indices.update([i, j])

        # 2. Cognitive component (personal best)
        for i in range(len(self.position)):
            if (
                r1 < c1
                and len(new_velocity) < self.max_velocity_size
                and self.position[i] != self.best_position[i]
            ):
                try:
                    j = self.best_position.index(self.position[i])
                    if can_add_swap(i, j):
                        new_velocity.append((i, j))
                        used_indices.update([i, j])
                except ValueError:
                    pass

        # 3. Social component (global best)
        for i in range(len(self.position)):
            if (
                r2 < c2
                and len(new_velocity) < self.max_velocity_size
                and self.position[i] != global_best_position[i]
            ):
                try:
                    j = global_best_position.index(self.position[i])
                    if can_add_swap(i, j):
                        new_velocity.append((i, j))
                        used_indices.update([i, j])
                except ValueError:
                    pass

        # 4. Mutation component
        if (
            random.random() < mutation_rate
            and len(new_velocity) < self.max_velocity_size
        ):
            attempts = 0
            while (
                attempts < MAX_ATTEMPTS_MUTATION
                and len(new_velocity) < self.max_velocity_size
            ):
                i, j = random.sample(range(len(self.position)), 2)
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
        if random.random() < mutation_rate:
            # Try block swaps (more powerful than single swaps)
            if random.random() < 0.3:  # 30% chance for block mutation
                block_size = random.randint(2, min(5, len(self.position) // 4))
                start = random.randint(0, len(self.position) - block_size)
                end = start + block_size
                valid = True
                for i in range(start, end):
                    for j in range(i + 1, end):
                        if self.position[i][0] == self.position[j][0]:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    # Reverse the block
                    self.position[start:end] = self.position[start:end][::-1]
            else:
                # Standard swap mutation
                attempts = 0
                while attempts < MAX_ATTEMPTS_MUTATION:
                    i, j = random.sample(range(len(self.position)), 2)
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
