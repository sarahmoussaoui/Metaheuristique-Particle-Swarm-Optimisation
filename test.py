import numpy as np
import random


class VerboseDiscretePSO:
    def __init__(
        self,
        objective_func,
        num_particles,
        num_dimensions,
        discrete_values,
        max_iter=100,
    ):
        self.objective_func = objective_func
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.discrete_values = discrete_values
        self.max_iter = max_iter

        # Initialize particles and velocities
        self.particles = np.zeros((num_particles, num_dimensions))
        self.velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))

        # Initialize personal bests
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(num_particles, np.inf)

        # Initialize global best
        self.global_best_position = None
        self.global_best_score = np.inf

        # PSO parameters
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
        self.w = 0.7  # inertia weight

        print(
            f"Initialized PSO with {num_particles} particles, {num_dimensions} dimensions"
        )
        print(f"Possible discrete values: {discrete_values}")

    def initialize_particles(self):
        print("\n=== Initializing Particles ===")
        for i in range(self.num_particles):
            for j in range(self.num_dimensions):
                self.particles[i, j] = random.choice(self.discrete_values)
            print(f"Particle {i}: {self.particles[i]} (random initial position)")

    def evaluate(self):
        print("\n=== Evaluating Particles ===")
        for i in range(self.num_particles):
            score = self.objective_func(self.particles[i])
            print(f"Particle {i}: Position {self.particles[i]} → Score: {score}")

            # Update personal best
            if score < self.personal_best_scores[i]:
                print(
                    f"  New personal best for particle {i}! (Previous: {self.personal_best_scores[i]}, New: {score})"
                )
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.particles[i].copy()

                # Update global best
                if score < self.global_best_score:
                    print(
                        f"  New global best! (Previous: {self.global_best_score}, New: {score})"
                    )
                    self.global_best_score = score
                    self.global_best_position = self.particles[i].copy()

    def update_velocities(self):
        print("\n=== Updating Velocities ===")
        for i in range(self.num_particles):
            print(f"\nParticle {i} current position: {self.particles[i]}")
            print(f"Current velocity: {self.velocities[i]}")
            print(f"Personal best: {self.personal_best_positions[i]}")
            print(f"Global best: {self.global_best_position}")

            for j in range(self.num_dimensions):
                r1, r2 = random.random(), random.random()
                cognitive = (
                    self.c1
                    * r1
                    * (self.personal_best_positions[i, j] - self.particles[i, j])
                )
                social = (
                    self.c2 * r2 * (self.global_best_position[j] - self.particles[i, j])
                )
                old_velocity = self.velocities[i, j]
                self.velocities[i, j] = (
                    self.w * self.velocities[i, j] + cognitive + social
                )

                print(f"  Dimension {j}:")
                print(f"    r1={r1:.3f}, r2={r2:.3f}")
                print(f"    cognitive={cognitive:.3f} (c1*r1*(pbest-current))")
                print(f"    social={social:.3f} (c2*r2*(gbest-current))")
                print(f"    velocity: {old_velocity:.3f} → {self.velocities[i,j]:.3f}")

    def update_positions(self):
        print("\n=== Updating Positions ===")
        for i in range(self.num_particles):
            print(f"\nParticle {i} current position: {self.particles[i]}")
            print(f"Current velocity: {self.velocities[i]}")

            for j in range(self.num_dimensions):
                # Apply sigmoid to velocity to get probability
                prob = 1 / (1 + np.exp(-self.velocities[i, j]))
                print(
                    f"  Dimension {j}: velocity={self.velocities[i,j]:.3f} → probability={prob:.3f}"
                )

                # Convert probability to discrete value
                if random.random() < prob:
                    choice = random.random()
                    if choice < 0.5:
                        new_val = self.personal_best_positions[i, j]
                        print(
                            f"    {choice:.3f} < 0.5 → move toward personal best ({new_val})"
                        )
                    else:
                        new_val = self.global_best_position[j]
                        print(
                            f"    {choice:.3f} ≥ 0.5 → move toward global best ({new_val})"
                        )
                else:
                    new_val = random.choice(self.discrete_values)
                    print(f"    random exploration → new value: {new_val}")

                self.particles[i, j] = new_val

            print(f"  New position: {self.particles[i]}")

    def optimize(self):
        print("=== Starting Optimization ===")
        self.initialize_particles()
        self.evaluate()  # Initial evaluation

        for iter in range(self.max_iter):
            print(f"\n\n=== Iteration {iter} ===")
            print(
                f"Current global best: {self.global_best_position} (score: {self.global_best_score})"
            )

            self.update_velocities()
            self.update_positions()
            self.evaluate()

            if iter % 10 == 0:
                print(f"\nIteration {iter} summary:")
                print(f"Best solution: {self.global_best_position}")
                print(f"Best score: {self.global_best_score}")

        print("\n=== Optimization Complete ===")
        return self.global_best_position, self.global_best_score


# Example Knapsack Problem Setup
weights = [2, 3, 4, 5, 9]
values = [3, 4, 5, 8, 10]
max_weight = 20


def knapsack_fitness(solution):
    total_value = np.sum(values * solution)
    total_weight = np.sum(weights * solution)

    if total_weight > max_weight:
        # Penalize solutions that exceed weight limit
        penalty = total_weight - max_weight
        print(
            f"  Invalid solution (weight {total_weight} > {max_weight}), applying penalty: {penalty}"
        )
        return total_weight  # We want to minimize this
    print(
        f"  Valid solution (weight {total_weight} ≤ {max_weight}), value: {total_value}"
    )
    return -total_value  # We want to maximize value (minimize negative value)


# Run the optimization with verbose output
print("=== Knapsack Problem ===")
print(f"Weights: {weights}")
print(f"Values: {values}")
print(f"Max weight: {max_weight}")

pso = VerboseDiscretePSO(
    knapsack_fitness,
    num_particles=3,  # Using fewer particles for clearer output
    num_dimensions=len(weights),
    discrete_values=[0, 1],
    max_iter=5,
)  # Fewer iterations for demonstration

best_solution, best_score = pso.optimize()

print("\n=== Final Results ===")
print(f"Best Solution: {best_solution}")
print(f"Selected items: {np.where(best_solution == 1)[0]}")
print(f"Total value: {-best_score}")  # Remember we used negative value
print(f"Total weight: {np.sum(weights * best_solution)}")
