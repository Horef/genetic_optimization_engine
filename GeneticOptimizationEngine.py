import os
import multiprocessing
import numpy as np
from time import time

from multiprocessing_helpers import DillProcess

class GOE:
    default_parameter_values = {int: 0, float: 0.0, bool: False}

    def __init__(self, fitness_function: callable,
                 parameter_types: dict,
                 maximize: bool = True,
                 parameter_mutations: dict = None,
                 projections: dict = None,
                 initial_set: dict = None,
                 scaler: callable = None,
                 mutation_probability: float = 0.1, crossover_probability: float = 0.1, dr: float = 0.01,
                 variability: float = 10,
                 num_generations: int = 100, num_agents: int = 100, num_workers: int = -1, 
                 print_progress: bool = False,
                 log_dir: str = None,
                 log_file: str = None,
                 seed: int = None):
        """
        Used to initialize the Genetic Optimization Engine.

        :param fitness_function: function to evaluate the fitness of the agents. It should accept a single parameter,
        a dictionary of the form {parameter_name: value} for each parameter.

        :param parameter_types: dictionary of form: {parameter_name: type}, where type is one of 'int', 'float', 'bool'
        or 'choice'. The 'choice' type should be used when the parameter can take a finite and discrete set of values.
        If possible, one of the other types should be used, as the 'choice' type is less efficient.

        :param maximize: if True, the genetic algorithm will try to maximize the fitness function, otherwise it will
        try to minimize it. True by default.

        :param parameter_mutations: dictionary of functions to use when mutating parameters. If None, defaults to mutation with
        a normal distribution with mean of previous value, and standard deviation of 25% of the previous value

        :param seed: seed value in case you want to reproduce the results

        :param projections: if the parameters need to be in a certain range, this dictionary should contain functions that
        will be applied to the parameters before evaluation. If None, defaults to identity function

        :param initial_set: dictionary of functions to randomly initiate the parameters. If None,
        uses mutation upon default values for each parameter.

        :param scaler: function to scale the fitness of the agents.
        If None, scales the fitness to be positive, and such that the minimal value will be zero,
        also adds a scale value, which is the difference between the maximal and the average fitness,
        divided by the number of agents.
        Additionally, scaler flips the fitness values, in case the genetic algorithm is set to minimize the fitness.

        :param mutation_probability: probability of mutation for each parameter.
        :param crossover_probability: probability of crossover between two chosen agents.
        :param dr: decay rate of the variability of the mutation. The variability is multiplied by (1 - dr) after each
        generation.
        :param variability: initial variability of the mutation. The mutation is a normal distribution with mean of the
        previous value, and standard deviation of variability * value.
        :param num_generations: number of epochs to run the genetic algorithm
        :param num_agents: number of agents to train for each epoch
        :param num_workers: number of workers to use for parallel processing. If -1, defaults to number of cores
        :param print_progress: if True, prints the progress of the genetic algorithm
        :param log_dir: directory to save the logs to. If None, does not save logs
        :param log_file: file to save the logs to. If None, but log_dir is not None, saves to 'goe_log.txt'
        """
        if seed:
            np.random.seed(seed)

        self.fitness_function = fitness_function
        self.maximize = maximize
        self.parameter_types = parameter_types
        # type checking
        for key in self.parameter_types.keys():
            if self.parameter_types[key] not in self.default_parameter_values.keys():
                raise ValueError(f'Parameter type {self.parameter_types[key]} is not supported')

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.variability = variability
        self.dr = dr

        # setting the mutation operators
        if parameter_mutations is None:
            self.parameter_mutations = {}
        else:
            self.parameter_mutations = parameter_mutations
        for key in self.parameter_types.keys():
            if key not in self.parameter_mutations:
                self.parameter_mutations[key] = self._default_mutation

        # setting the projection operators
        if projections is None:
            self.projections = {}
        else:
            self.projections = projections
        for key in self.parameter_types.keys():
            if key not in self.projections:
                self.projections[key] = self._identity_projection

        # setting the scaler function
        if scaler is None:
            self.scaler = self._scaler
        else:
            self.scaler = scaler

        # setting the initial set
        if initial_set is None:
            self.initial_set = {}
        else:
            self.initial_set = initial_set
        for key in self.parameter_types.keys():
            if key not in self.initial_set:
                self.initial_set[key] = self.default_parameter_values[self.parameter_types[key]]

        # initializing the log file, if needed
        self.log_file = None
        if log_dir is not None:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if log_file is None:
                log_file = 'goe_log.txt'
            self.log_file = os.path.join(log_dir, log_file)

        self.num_generations = num_generations
        self.num_agents = num_agents
        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()

        # initializing the first set of agents
        self.agents = [self._mutate(self.initial_set, mutation_probability=1) for _ in range(self.num_agents)]

        if print_progress:
            print('Initialization done, evaluating initial agents...')
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write('Initialization done, evaluating initial agents...\n')

        start_time = time()
        if self.num_workers > 1:
            max_workers = min(self.num_workers, self.num_agents)
            index_splits = np.array_split(range(self.num_agents), max_workers)

            with multiprocessing.Manager() as manager:
                agents = manager.list(self.agents)
                fitness = manager.list([0 for _ in range(self.num_agents)])

                workers = []
                for j in range(max_workers):
                    worker = DillProcess(target=self._evaluate_parallel, args=(agents, fitness, index_splits[j]))
                    workers.append(worker)
                    worker.start()

                for worker in workers:
                    worker.join()

                self.agents = list(agents)
                self.fitness = list(fitness)
        else:
            self.fitness = [self._evaluate(agent) for agent in self.agents]

        end_time = time()
        if print_progress:
            print(f'Initialization done, took {end_time - start_time:.2f} seconds')
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f'Initialization done, took {end_time - start_time:.2f} seconds\n')
                for agent, fitness in zip(self.agents, self.fitness):
                    f.write(f'Agent: {agent}, Fitness: {fitness}\n')
                f.write('\n\n')

        if print_progress:
            print('Starting the genetic algorithm...')
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write('Starting the genetic algorithm...\n')

        for i in range(self.num_generations):
            start_time = time()

            self.variability = self.variability * (1 - self.dr)
            fitness_variability = np.std(self.fitness)
            self.variability = min(self.variability, fitness_variability)

            self.agents = self._reproduce(self.agents, self.fitness)
            self.agents = [self._mutate(agent) for agent in self.agents]

            if self.num_workers > 1:
                max_workers = min(self.num_workers, self.num_agents)
                index_splits = np.array_split(range(self.num_agents), max_workers)

                with multiprocessing.Manager() as manager:
                    agents = manager.list(self.agents)
                    fitness = manager.list(self.fitness)

                    workers = []
                    for j in range(max_workers):
                        worker = DillProcess(target=self._evaluate_parallel, args=(agents, fitness, index_splits[j]))
                        workers.append(worker)
                        worker.start()

                    for worker in workers:
                        worker.join()

                    self.agents = list(agents)
                    self.fitness = list(fitness)

            else:
                self.fitness = [self._evaluate(agent) for agent in self.agents]

            end_time = time()
            if self.maximize:
                self.best_agent = self.agents[np.argmax(self.fitness)]
                self.best_fitness = np.max(self.fitness)
            else:
                self.best_agent = self.agents[np.argmin(self.fitness)]
                self.best_fitness = np.min(self.fitness)

            if print_progress:
                print(f'Generation {i + 1}/{self.num_generations} done, best fitness: {self.best_fitness}, '
                      f'best agent: {self.best_agent}\n'
                      f'It took {end_time - start_time:.2f} seconds\n')
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(f'Generation {i + 1}/{self.num_generations} done, best fitness: {self.best_fitness}, '
                            f'best agent: {self.best_agent}\n'
                            f'It took {end_time - start_time:.2f} seconds\n')
                    for agent, fitness in zip(self.agents, self.fitness):
                        f.write(f'Agent: {agent}, Fitness: {fitness}\n')
                    f.write('\n\n')

        if self.maximize:
            self.best_agent = self.agents[np.argmax(self.fitness)]
            self.best_fitness = np.max(self.fitness)
        else:
            self.best_agent = self.agents[np.argmin(self.fitness)]
            self.best_fitness = np.min(self.fitness)

    @staticmethod
    def _identity_projection(value):
        return value

    def _default_mutation(self, parameter_type, value):
        if parameter_type == int:
            return int(np.random.normal(value, self.variability))
        elif parameter_type == float:
            return np.random.normal(value, self.variability)
        elif parameter_type == 'choice':
            return np.random.choice(self.initial_set[parameter_type])
        else:
            # the only way to mutate a boolean is to flip it
            return not value

    def _mutate(self, agent, mutation_probability=None):
        if mutation_probability is None:
            mutation_probability = self.mutation_probability
        new_agent = agent.copy()
        for key in agent.keys():
            if np.random.rand() < mutation_probability:
                new_agent[key] = self.parameter_mutations[key](self.parameter_types[key], agent[key])
        return new_agent

    def _crossover(self, agent1, agent2):
        new_agent1 = {}
        new_agent2 = {}

        if np.random.rand() < self.crossover_probability:
            # selecting the random crossover point
            crossover_point = np.random.randint(0, len(agent1))

            for i, key in enumerate(agent1.keys()):
                if i < crossover_point:
                    new_agent1[key] = agent1[key]
                    new_agent2[key] = agent2[key]
                else:
                    new_agent1[key] = agent2[key]
                    new_agent2[key] = agent1[key]

        else:
            new_agent1 = agent1
            new_agent2 = agent2

        return new_agent1, new_agent2

    def _scaler(self, values: list) -> list:
        values = np.array(values)
        # flipping the values, if the genetic algorithm is set to minimize the fitness
        if not self.maximize:
            values = -values
        # finding the average value
        scale = (np.max(values)-np.mean(values))/len(values)
        if scale == 0:
            scale = 1
        # adding the minimal value and the scale to each value
        min_val = np.min(values)
        values = (values - min_val + scale)
        return values.tolist()

    def _select(self, agents, fitness):
        # rescaling the fitness to be positive and to give each fitness a chance to be selected
        fitness = self.scaler(fitness)
        # normalizing the fitness to be in the range [0, 1], to represent probabilities
        # copy the fitness to avoid changing the original list
        fitness = np.array(fitness)
        fitness = fitness / np.sum(fitness)

        # randomly selecting two agents based on their fitness
        best_indices = np.random.choice(range(len(agents)), size=2, p=fitness)

        return agents[best_indices[0]], agents[best_indices[1]]

    def _evaluate_parallel(self, agents, fitness, indices):
        for i in indices:
            fitness[i] = self._evaluate(agents[i])

    def _evaluate(self, agent):
        projected_agent = {key: self.projections[key](value) for key, value in agent.items()}
        return self.fitness_function(projected_agent)

    def _reproduce(self, agents, fitness):
        new_agents = []
        while len(new_agents) < self.num_agents:
            agent1, agent2 = self._select(agents, fitness)
            new_agent1, new_agent2 = self._crossover(agent1, agent2)
            new_agents.append(new_agent1)
            new_agents.append(new_agent2)
        if len(new_agents) > self.num_agents:
            # if the number of new agents is odd, we need to remove one
            new_agents = new_agents[:self.num_agents]
        return new_agents

    def get_best_agent(self) -> (dict, float):
        return (self.best_agent, self.best_fitness)

def fitness_wrapper(func: callable) -> callable:
    """
    Used to wrap a generic function to be used as a fitness function for the Genetic Optimization Engine.

    :param func: function to evaluate the fitness of the agents. All of its parameters should be keyword arguments.
    :return: function that receives a dictionary of the form {parameter_name: value} for each parameter.
    And calls the original function with the parameters as keyword arguments.
    """

    def fitness(parameters: dict) -> float:
        return func(**parameters)

    return fitness