import numpy as np
from deap import base, creator, gp, tools, algorithms
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# Define Boolean n-ary AND, OR, and NOT operations
def n_ary_and(*args):
    return all(args)

def n_ary_or(*args):
    return any(args)

def not_op(x):
    return not x

# Define the primitive set for Boolean GP
pset = gp.PrimitiveSet("MAIN", 3)  # 3 boolean inputs for simplicity
pset.addPrimitive(n_ary_and, 2)    # 2-ary AND
pset.addPrimitive(n_ary_or, 2)     # 2-ary OR
pset.addPrimitive(not_op, 1)       # Unary NOT
pset.addTerminal(True)
pset.addTerminal(False)

# Create a custom fitness function
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initialize the toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the compile method to convert individual trees to callable functions
toolbox.register("compile", gp.compile, pset=pset)

# Target Boolean function (e.g., 3-input XOR)
def target_boolean_function(x1, x2, x3):
    return (x1 and not x2 and not x3) or (not x1 and x2 and not x3) or (not x1 and not x2 and x3)

# Define the fitness function
def eval_fitness(individual):
    # Compile the individual into a callable function
    func = toolbox.compile(expr=individual)
    
    # Define all possible inputs for a 3-variable Boolean function
    inputs = [(x1, x2, x3) for x1 in [False, True] for x2 in [False, True] for x3 in [False, True]]
    
    # Calculate the fitness score (how well the individual matches the target function)
    correct_outputs = [target_boolean_function(*inp) for inp in inputs]
    individual_outputs = [func(*inp) for inp in inputs]
    
    # Fitness is the number of correct matches (maximize this)
    fitness = sum([int(correct == output) for correct, output in zip(correct_outputs, individual_outputs)])
    
    return fitness,

toolbox.register("evaluate", eval_fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)


# Bayesian Optimization setup for finding tree depth
search_space = [Integer(1, 5, name='depth')]

@use_named_args(search_space)
def bo_objective(depth):
    # Build an individual with the specified depth
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=depth, max_=depth)
    individual = toolbox.individual()
    
    # Evaluate the individual's fitness
    fitness = toolbox.evaluate(individual)[0]
    return -fitness  # We want to maximize fitness, so minimize the negative fitness

# Perform Bayesian optimization
res = gp_minimize(bo_objective, search_space, n_calls=30, random_state=0)

print(f"Best tree depth: {res.x[0]}")
print(f"Best fitness: {-res.fun}")

# Optionally, evolve a population after finding good depth (for experimentation)
def evolve_population():
    population = toolbox.population(n=100)
    hall_of_fame = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Run evolution
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, 
                        stats=stats, halloffame=hall_of_fame, verbose=True)
    return hall_of_fame[0]

best_individual = evolve_population()
print(f"Best evolved individual: {best_individual}")
