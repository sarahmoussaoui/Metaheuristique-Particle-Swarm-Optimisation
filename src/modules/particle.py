from copy import deepcopy
from typing import List, Tuple
import random


class Particle:
    """Represents a particle in the PSO algorithm with verbose printing."""

    def __init__(
        self, sequence: List[Tuple[int, int]], job_machine_dict: dict[int, list[int]]
    ):
        self.position = sequence
        self.velocity = []
        self.best_position = deepcopy(sequence)
        self.best_fitness = float("inf")
        self.fitness = float("inf")
        self.job_machine_dict = job_machine_dict
        self.initialize_velocity()
        print(f"\nParticle initialized:")
        print(f"Initial position: {self.position}")
        print(f"Initial velocity: {self.velocity}")
        print(f"Initial best position: {self.best_position}")

    def initialize_velocity(self, max_swaps: int = None):
        """Initialize velocity with random swaps that respect all constraints.

        Args:
           max_swaps: Maximum number of swaps to generate. If None, uses random number up to half length.
        """
        if max_swaps is None:
            max_swaps = random.randint(0, max(1, len(self.position)))

        self.velocity = []
        used_indices = set()
        attempts = 0
        max_attempts = 100  # Prevent infinite loops

        print(f"\nInitializing velocity with max {max_swaps} swaps...")
        print(f"Current position: {self.position}")

        while len(self.velocity) < max_swaps and attempts < max_attempts:
            attempts += 1

            # Randomly select two distinct indices
            i, j = random.sample(range(len(self.position)), 2)
            print(f"\nAttempt {attempts}: Considering swap between indices {i} and {j}")
            print(f"Operations: {self.position[i]} and {self.position[j]}")

            # Check basic constraints first
            if self.position[i][0] == self.position[j][0]:
                print("× Rejected - same job")
                continue

            if i in used_indices or j in used_indices:
                print("× Rejected - index already used")
                continue

            # Create temp position to validate constraints
            temp_position = deepcopy(self.position)
            temp_position[i], temp_position[j] = temp_position[j], temp_position[i]

            # Check machine order for both affected jobs
            job_i = temp_position[i][0]
            job_j = temp_position[j][0]

            valid_i = is_machine_order_valid(
                temp_position, job_i, self.job_machine_dict
            )
            valid_j = is_machine_order_valid(
                temp_position, job_j, self.job_machine_dict
            )

            if valid_i and valid_j:
                self.velocity.append((i, j))
                used_indices.update([i, j])
                print(f"✓ Added valid swap: positions {i} and {j}")
                print(f"New temporary position after swap: {temp_position}")
            else:
                print(f"× Rejected - would violate machine order:")
                if not valid_i:
                    print(f"  - Invalid order for job {job_i}")
                if not valid_j:
                    print(f"  - Invalid order for job {job_j}")

        print(f"\nFinal initialized velocity: {self.velocity}")
        print(f"Total valid swaps found: {len(self.velocity)}/{max_swaps}")

    def update_position(self):
        """Update position by applying velocity (swaps), skipping invalid ones and preserving machine order."""
        print(f"\nUpdating position for particle:")
        print(f"Current position: {self.position}")
        print(f"Current velocity (swaps to apply): {self.velocity}")

        new_position = deepcopy(self.position)
        for i, j in self.velocity:
            if i < len(new_position) and j < len(new_position):
                print(f"Applying swap: positions {i} and {j}")
                print(f"Before swap: {new_position[i]} <-> {new_position[j]}")
                new_position[i], new_position[j] = new_position[j], new_position[i]
                print(f"After swap: {new_position[i]} <-> {new_position[j]}")

        self.position = new_position
        self.velocity = self.velocity[: len(new_position)]  # Trim if needed
        print(f"New position after update: {self.position}")
        print(f"Velocity after trimming: {self.velocity}")

    def update_velocity(
        self,
        global_best_position: List[Tuple[int, int]],
        w: float, 
        c1: float,
        c2: float,
    ):
        """Update velocity based on personal and global best positions with constraints."""
        print(f"\nUpdating velocity for particle:")
        print(f"Current position: {self.position}")
        print(f"Current velocity: {self.velocity}")
        print(f"Personal best position: {self.best_position}")
        print(f"Global best position: {global_best_position}")

        new_velocity = []
        used_indices = set()

        def can_add_swap(i, j):
            return (
                i not in used_indices
                and j not in used_indices
                and self.position[i][0] != self.position[j][0]  # different jobs
            )

        # Inertia component
        print("\nApplying inertia component (previous velocity):")
        for swap in self.velocity:
            i, j = swap
            if random.random() < w:
                print(f"Considering keeping swap {i},{j} (probability {w})")
                if can_add_swap(i, j):
                    # Simulate swap and check constraints
                    new_position = self.position.copy()
                    new_position[i], new_position[j] = new_position[j], new_position[i]
                    print(
                        f"Tentative swap: {i},{j} -> {new_position[i]} <-> {new_position[j]}"
                    )

                    if is_machine_order_valid(
                        new_position, new_position[i][0], self.job_machine_dict
                    ) and is_machine_order_valid(
                        new_position, new_position[j][0], self.job_machine_dict
                    ):
                        new_velocity.append((i, j))
                        used_indices.update([i, j])
                        print(
                            f"Added swap {i},{j} to new velocity (machine order valid)"
                        )
                    else:
                        print(f"Skipping swap {i},{j} due to invalid machine order")
                else:
                    print(f"Skipping swap {i},{j} (indices already used or same job)")
            else:
                print(f"Dropping swap {i},{j} (random check failed)")

        # Cognitive component
        print("\nApplying cognitive component (personal best):")
        for i in range(len(self.position)):
            if random.random() < random.random() * c1:
                print(f"\nCognitive component triggered for position {i}")
                if self.position[i] != self.best_position[i]:
                    j = self.best_position.index(self.position[i])
                    print(
                        f"Current op at {i}: {self.position[i]}, in personal best at {j}"
                    )
                    if can_add_swap(i, j):
                        new_position = self.position.copy()
                        new_position[i], new_position[j] = (
                            new_position[j],
                            new_position[i],
                        )
                        print(
                            f"Tentative swap: {i},{j} -> {new_position[i]} <-> {new_position[j]}"
                        )

                        if is_machine_order_valid(
                            new_position, new_position[i][0], self.job_machine_dict
                        ) and is_machine_order_valid(
                            new_position, new_position[j][0], self.job_machine_dict
                        ):
                            new_velocity.append((i, j))
                            used_indices.update([i, j])
                            print(f"Added cognitive swap {i},{j} to new velocity")
                        else:
                            print(
                                f"Skipping cognitive swap {i},{j} (invalid machine order)"
                            )
                    else:
                        print(
                            f"Skipping cognitive swap {i},{j} (indices already used or same job)"
                        )
                else:
                    print(f"No swap needed at {i} (matches personal best)")
            else:
                print(
                    f"Cognitive component not triggered for position {i} (random check)"
                )

        # Social component
        print("\nApplying social component (global best):")
        for i in range(len(self.position)):
            if random.random() < random.random() * c2: # Probability random() * c2 controls swarm influence
                print(f"\nSocial component triggered for position {i}")
                if self.position[i] != global_best_position[i]:
                    j = global_best_position.index(self.position[i])
                    print(
                        f"Current op at {i}: {self.position[i]}, in global best at {j}"
                    )
                    if can_add_swap(i, j):
                        new_position = self.position.copy()
                        new_position[i], new_position[j] = (
                            new_position[j],
                            new_position[i],
                        )
                        print(
                            f"Tentative swap: {i},{j} -> {new_position[i]} <-> {new_position[j]}"
                        )

                        if is_machine_order_valid(
                            new_position, new_position[i][0], self.job_machine_dict
                        ) and is_machine_order_valid(
                            new_position, new_position[j][0], self.job_machine_dict
                        ):
                            new_velocity.append((i, j))
                            used_indices.update([i, j])
                            print(f"Added social swap {i},{j} to new velocity")
                        else:
                            print(
                                f"Skipping social swap {i},{j} (invalid machine order)"
                            )
                    else:
                        print(
                            f"Skipping social swap {i},{j} (indices already used or same job)"
                        )
                else:
                    print(f"No swap needed at {i} (matches global best)")
            else:
                print(f"Social component not triggered for position {i} (random check)")

        self.velocity = new_velocity
        print(f"\nFinal new velocity: {self.velocity}")


def is_machine_order_valid(
    pos: int, job_id: int, job_machine_dict: dict[int, list[int]]
):
    machines_in_position = [op[1] for op in pos if op[0] == job_id]
    correct_order = job_machine_dict[job_id]
    valid = all(m1 == m2 for m1, m2 in zip(machines_in_position, correct_order))
    if not valid:
        print(
            f"Machine order would be invalid for job {job_id} with sequence {machines_in_position} (should be {correct_order})"
        )
    return valid
