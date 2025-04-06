import random


class Individual:
    def __init__(self, mutation_rate, chromosome, fitness_func: callable):
        self.chromosome_length = len(chromosome)
        self.chromosome = chromosome
        self._calculate_fitness = fitness_func
        self.fitness = self._calculate_fitness(chromosome)
        self.mutation_rate = mutation_rate

    def calculate_fitness(self):
        return self._calculate_fitness(self.chromosome)

    def _random_chromosome(self):
        return [random.randint(0, 1) for _ in range(self.chromosome_length)]

    def mutate(self):


    def crossover(self, partner):


    def _select_parents(self):
        # Tournament selection
        tournament = random.sample(self.population, k=4)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0], tournament[1]
