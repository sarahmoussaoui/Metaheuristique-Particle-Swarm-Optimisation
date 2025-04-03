import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Particle:
    def __init__(self, dim, minx, maxx):
        self.position = np.random.uniform(minx, maxx, dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = np.copy(self.position)
        self.best_value = float("inf")


class PSO:
    def __init__(
        self,
        objective_func,
        dim,
        n_particles,
        max_iter,
        minx,
        maxx,
        w=0.7,
        c1=1.4,
        c2=1.4,
    ):
        self.objective_func = objective_func
        self.dim = dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.minx = minx
        self.maxx = maxx
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient

        self.particles = [Particle(dim, minx, maxx) for _ in range(n_particles)]
        self.global_best_position = np.random.uniform(minx, maxx, dim)
        self.global_best_value = float("inf")

        self.best_values_history = []

    def optimize(self):
        for _ in range(self.max_iter):
            for particle in self.particles:
                # Evaluate current fitness
                current_value = self.objective_func(particle.position)

                # Update personal best
                if current_value < particle.best_value:
                    particle.best_value = current_value
                    particle.best_position = particle.position.copy()

                # Update global best
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = particle.position.copy()

            # Update velocities and positions
            for particle in self.particles:
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                # Update velocity
                cognitive_velocity = (
                    self.c1 * r1 * (particle.best_position - particle.position)
                )
                social_velocity = (
                    self.c2 * r2 * (self.global_best_position - particle.position)
                )
                particle.velocity = (
                    self.w * particle.velocity + cognitive_velocity + social_velocity
                )

                # Update position
                particle.position += particle.velocity

                # Apply position bounds
                particle.position = np.clip(particle.position, self.minx, self.maxx)

            self.best_values_history.append(self.global_best_value)

        return self.global_best_position, self.global_best_value

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_values_history)
        plt.title("PSO Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Best Value")
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Define an objective function (e.g., Sphere function)
    def sphere_function(x):
        return np.sum(x**2)

    # Parameters
    dim = 2  # problem dimension
    n_particles = 30
    max_iter = 100
    minx, maxx = -5, 5

    # Run PSO
    pso = PSO(sphere_function, dim, n_particles, max_iter, minx, maxx)
    best_position, best_value = pso.optimize()

    print(f"Best position: {best_position}")
    print(f"Best value: {best_value}")

    # Plot convergence
    pso.plot_convergence()

    # Optional: Visualize the search space (for 2D problems)
    if dim == 2:
        x = np.linspace(minx, maxx, 100)
        y = np.linspace(minx, maxx, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = sphere_function(np.array([X[i, j], Y[i, j]]))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)

        # Plot particles' final positions
        for particle in pso.particles:
            ax.scatter(
                particle.position[0],
                particle.position[1],
                sphere_function(particle.position),
                color="red",
            )

        ax.scatter(
            best_position[0],
            best_position[1],
            best_value,
            color="green",
            s=100,
            marker="*",
            label="Global Best",
        )
        ax.set_title("PSO Optimization on Sphere Function")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()
