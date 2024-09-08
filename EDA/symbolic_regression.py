import random
import numpy as np

# Constants for the algorithm
POPULATION_SIZE = 100  # R, total population size
NUM_FEATURES = 5       # n, number of variables/features
NUM_SELECTED = 20      # N, selected individuals for modeling
NUM_GENERATIONS = 50   # Number of iterations
MUTATION_RATE = 0.1    # Probability of mutation

# Fitness function (example: mean squared error)
def fitness_function(individual, target_func):
    error = np.sum((np.array(individual) - target_func) ** 2)
    return 1 / (1 + error)  # Inverse of error, higher is better

# Initialize population D0 with random individuals
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = np.random.randint(1, 10, size=NUM_FEATURES)  # Random values from 1 to 9
        print(individual)
        population.append(individual)
    return population

# Select N individuals based on their fitness (higher fitness = more likely to be selected)
def selection(population, fitness_scores):
    selected_indices = np.argsort(fitness_scores)[-NUM_SELECTED:]  # Select top N individuals
    return [population[i] for i in selected_indices]

# Create a probabilistic model by estimating feature distributions
def induce_probability_model(selected_population):
    # For each feature, calculate the probability distribution based on selected individuals
    feature_probabilities = []
    selected_population = np.array(selected_population)
    
    for i in range(NUM_FEATURES):
        # Get values for this feature across all selected individuals
        values, counts = np.unique(selected_population[:, i], return_counts=True)
        probabilities = counts / np.sum(counts)
        feature_probabilities.append((values, probabilities))
    
    return feature_probabilities

# Sample new individuals from the probability model
def sample_from_probability_model(probability_model):
    new_population = []
    for _ in range(POPULATION_SIZE):
        individual = []
        for values, probabilities in probability_model:
            sampled_value = np.random.choice(values, p=probabilities)
            individual.append(sampled_value)
        new_population.append(individual)
    return new_population
def introduce_mutation(population, mutation_rate=0.1):
    mutated_population = []
    for individual in population:
        mutated_individual = individual.copy()  # Copy to avoid mutating the original individual
        if random.random() < mutation_rate:
            # Select a random position (gene) to mutate
            mutation_index = random.randint(0, NUM_FEATURES - 1)
            # Replace it with a new random value (within a valid range, say 1 to 9)
            mutated_individual[mutation_index] = random.randint(1, 9)
        mutated_population.append(mutated_individual)
    return mutated_population
# Main function to run the algorithm
def run_symbolic_regression(target_func):
    population = initialize_population()
    
    for generation in range(NUM_GENERATIONS):
        # Calculate fitness for the entire population
        fitness_scores = [fitness_function(ind, target_func) for ind in population]
        
        # Selection: Pick the best N individuals
        selected_population = selection(population, fitness_scores)
        
        # Induce probability model based on selected individuals
        probability_model = induce_probability_model(selected_population)
        
        # Sample new population based on the probability model
        population = sample_from_probability_model(probability_model)
        
        # Optionally introduce mutation (mutation rate should be low)
        if random.random() < MUTATION_RATE:
            population = introduce_mutation(population)
        
        # Check progress or stopping condition
        best_individual = population[np.argmax(fitness_scores)]
        best_score = max(fitness_scores)
        print(f'Generation {generation}: Best Score = {best_score}')
        
        # Stopping criterion can be added here (e.g., fitness threshold)
    
    # Final result
    best_individual = population[np.argmax(fitness_scores)]
    return best_individual

# Example: A target function for symbolic regression (adjust this to your target)
target_function = np.array([3, 5, 2, 8, 1])

# Run the symbolic regression algorithm
best_solution = run_symbolic_regression(target_function)
print(f'Best Solution: {best_solution}')
