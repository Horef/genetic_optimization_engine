import numpy as np

from GeneticOptimizationEngine import GOE
from GeneticOptimizationEngine import fitness_wrapper

def func1(x: float, y: float):
    return -x**2 - y**2 + 10

def func2(x: float, y: float):
    return -x**4 +3*x**3 - y**4 + y**2

def func3(x: float, y: float):
    return np.abs(np.floor(x**2) + x + np.floor(y**2) - y)

if __name__ == '__main__':
    print('Testing GOE with func1')
    goe = GOE(fitness_function=fitness_wrapper(func1), parameter_types={'x': float, 'y': float},
              maximize=True, initial_set={'x': 100, 'y': 100}, num_workers=1, num_generations=300, num_agents=100,
              variability=10, dr=0.01, seed=3, log_dir='./')
    goe.initialize_agents()
    goe.run_evolution()
    best_agent, best_fitness = goe.get_best_agent()
    print(f'Best agent: {best_agent}, best fitness: {best_fitness}')

    print('Testing GOE with func2')
    goe = GOE(fitness_function=fitness_wrapper(func2), parameter_types={'x': float, 'y': float},
              maximize=True, initial_set={'x': 100, 'y': 100}, num_workers=1, num_generations=300, num_agents=100,
              variability=10, dr=0.01, seed=3, log_dir='./')
    goe.initialize_agents()
    goe.run_evolution()
    best_agent, best_fitness = goe.get_best_agent()
    print(f'Best agent: {best_agent}, best fitness: {best_fitness}')

    print('Testing GOE with func3')
    goe = GOE(fitness_function=fitness_wrapper(func3), parameter_types={'x': float, 'y': float},
              maximize=False, initial_set={'x': 100, 'y': 100}, num_workers=1, num_generations=300, num_agents=100,
              variability=10, dr=0.01, seed=3, log_dir='./')
    goe.initialize_agents()
    goe.run_evolution()
    best_agent, best_fitness = goe.get_best_agent()
    print(f'Best agent: {best_agent}, best fitness: {best_fitness}')