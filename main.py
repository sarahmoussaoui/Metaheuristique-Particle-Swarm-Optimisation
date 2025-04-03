from src.modules.modelisation import JSSP
from src.modules.visualisation import ScheduleVisualizer
from src.modules.pso import PSOOptimizer

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
    print(jssp)

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
