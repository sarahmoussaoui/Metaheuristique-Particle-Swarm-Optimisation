from src.modules.individual import Individual
from copy import deepcopy
from typing import List, Tuple
import random


class GeneticAlgorithm:
    def __init__(self, jssp):
        self.jssp = jssp
        self.iteration_history = []
        self.makespan_history = []
        self.population_size = None  # Will be set in evolve()

    def generate_initial_chromosome(self) -> List[Tuple[int, int]]:
        """Generates a valid initial sequence preserving operation order within jobs."""
        remaining_ops = deepcopy(self.jssp.job_machine_dict)
        chromosome = []

        while any(remaining_ops.values()):
            available_jobs = [j for j, ops in remaining_ops.items() if ops]
            job = random.choice(available_jobs)
            op_idx = remaining_ops[job].pop(0)
            chromosome.append((job, op_idx))

        return chromosome

    def _select_parents(self):
        """Select parents using tournament selection."""
        tournament_size = max(2, len(self.population) // 5)
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0], tournament[1]

    def evolve(self, mutation_rate, population_size, generations):
        """Run the genetic algorithm and return the best solution."""
        self.population_size = population_size
        self.population = [
            Individual(
                mutation_rate=mutation_rate,
                chromosome=self.generate_initial_chromosome(),
                fitness_func=self.jssp.evaluate_schedule,
            )
            for _ in range(population_size)
        ]

        best_individual = None

        for generation in range(generations):
            # Evaluate population
            for individual in self.population:
                individual.calculate_fitness()

            # Sort by fitness (descending)
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            # Track best individual
            current_best = self.population[0]
            if (
                best_individual is None
                or current_best.fitness > best_individual.fitness
            ):
                best_individual = deepcopy(current_best)

            # Record history
            self.iteration_history.append(generation)
            self.makespan_history.append(
                1 / current_best.fitness
            )  # Assuming fitness is 1/makespan

            print(
                f"Generation {generation}: Best Fitness = {current_best.fitness}, Makespan = {1/current_best.fitness:.2f}"
            )

            # Create next generation
            next_generation = self.population[:2]  # Elitism: keep top 2

            while len(next_generation) < population_size:
                parent1, parent2 = self._select_parents()
                child1, child2 = parent1.crossover(parent2)
                child1.mutate()
                child2.mutate()
                next_generation += [child1, child2]

            self.population = next_generation

        # Return the best schedule and its makespan
        best_schedule = best_individual.chromosome
        best_makespan = 1 / best_individual.fitness
        return best_schedule, best_makespan
