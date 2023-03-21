import numpy as np
import random
import matplotlib.pyplot as plt


class Gene:
    def __init__(self, gene_length, max_age):
        self.gene_length = gene_length
        self.max_age = max_age
        self.age = 0
        self.gene = self.initialize_gene()

    def initialize_gene(self):
        # Initialize gene with random bits
        # gene = np.random.randint(2, size=self.gene_length) # Random bits initialization
        gene = np.zeros(self.gene_length, dtype=int)        # All zeros initialization
        return gene

    def survival_function(self, gene):
        # Remove genes based on age
        return gene.age < self.max_age


class GeneEnsemble(Gene):
    def __init__(self, gene_length, max_age, mutation_rate, recombination_rate, growth_rate, population_size, carrying_capacity, run_idx=0):
        super().__init__(gene_length, max_age)
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.growth_rate = growth_rate
        self.population_size = population_size
        self.carrying_capacity = carrying_capacity
        self.run_idx = run_idx
        self.population = self.initialize_population()
        self.generations = 0
        self.initial_gene_composition = np.mean([k.gene for k in self.population], axis=0)  # Calculate initial gene composition
        self.average_hamming_distances = []
        self.variance_hamming_distances = []
        self.population_sizes = []

    def mutate(self, gene):
        # Mutate a gene with a certain probability
        for i in range(self.gene_length):
            if np.random.random() < self.mutation_rate:
                gene[i] = 1 - gene[i]  # Flip the bit
        return gene

    def recombine(self, parent1, parent2):
        # Recombine two parent genes to create a new child gene
        child = np.zeros(self.gene_length, dtype=int)
        for i in range(self.gene_length):
            if np.random.random() < self.recombination_rate:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def initialize_population(self):
        # Initialize population with random genes
        population = []
        for i in range(self.population_size):
            gene = Gene(self.gene_length, self.max_age)
            population.append(gene)
        return population

    def selection(self):
        # Select parents for recombination
        parents = random.sample(self.population, k=2)
        return parents[0], parents[1]

    def run(self, num_generations):
        for idx in range(num_generations):

            # # Calculate average hamming distance
            # average_hamming_distance = np.mean(
            #     [np.sum(gene.gene != self.initial_gene_composition) for gene in self.population])
            # self.average_hamming_distances.append(average_hamming_distance)

            # Calculate variance of pairwise hamming distances
            pairwise_hamming_distances = []
            for jdx in range(len(self.population)):
                for kdx in range(jdx + 1, len(self.population)):
                    pairwise_hamming_distance = np.sum(self.population[jdx].gene != self.population[kdx].gene)
                    pairwise_hamming_distances.append(pairwise_hamming_distance)
            variance_pairwise_hamming_distances = np.var(pairwise_hamming_distances)
            self.variance_hamming_distances.append(variance_pairwise_hamming_distances)

            # Update population size based on logistic growth
            if int(self.carrying_capacity) == int(self.population_size):
                new_population_size = self.population_size
            sigmoid_midpoint = np.log(self.carrying_capacity / self.population_size - 1) / self.growth_rate
            new_population_size = int(
                self.carrying_capacity / (1 + np.exp(-self.growth_rate * (self.generations - sigmoid_midpoint))))
            self.population_sizes.append(new_population_size)

            # Create new generation
            new_population = []
            for _ in range(new_population_size):
                parent1, parent2 = self.selection()
                child = Gene(self.gene_length, self.max_age)
                child_gene = GeneEnsemble.recombine(self, parent1.gene, parent2.gene)
                child_gene = GeneEnsemble.mutate(self, child_gene)
                child.gene = child_gene
                new_population.append(child)

            # Keep old genes based on survival function
            survivors = []
            for gene in self.population:
                gene.age += 1
                if gene.survival_function(gene):
                    survivors.append(gene)

            # Combine new and old populations
            self.population = survivors + new_population

            # Increment generation
            self.generations += 1

        # # Plot average hamming distance over time
        # plt.errorbar(range(1, num_generations + 1), self.average_hamming_distances, np.sqrt(np.array(self.variance_hamming_distances)))
        # plt.savefig(str(self.run_idx) + "_hamming.png")
        # plt.xlabel("Generation")
        # plt.ylabel("Average Hamming Distance")
        # plt.title(str(self.run_idx) + " Hamming plot")
        # plt.show()

        # # Plot population size over time
        # plt.plot(range(1, num_generations + 1), self.population_sizes)
        # plt.savefig(str(self.run_idx) + "_population.png")
        # plt.xlabel("Generation")
        # plt.ylabel("Population Size")
        # plt.title("Population plot")
        # plt.show()

        return self.variance_hamming_distances
