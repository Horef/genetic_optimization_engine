import os
import multiprocessing
import numpy as np
from time import time

from multiprocessing_helpers import DillProcess

class GOE:
    default_parameter_values = {int: 0, float: 0.0, bool: False, list: [0.0], 'choice': [0]}

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
            If 'choice' is used, the initial_set should contain a list of possible values for that parameter.

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

        self.flags = {'maximize': maximize}

        self.fitness_function = fitness_function
        self.parameter_types = parameter_types
        # type checking
        for key in self.parameter_types.keys():
            if self.parameter_types[key] not in self.default_parameter_values.keys():
                raise ValueError(f'Parameter type {self.parameter_types[key]} is not supported')

        # setting the order of the parameters to make sure that the crossover is consistent
        self.parameter_order = list(enumerate(self.parameter_types.keys()))
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

        self.print_progress = print_progress
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

        self.agents = None
        self.fitness = None

        self.best_agent = None
        self.last_best_agent = None
        self.best_fitness = None
        self.last_best_fitness = None

    def initialize_agents(self):
        """
        Initializes the agents for the genetic algorithm.
        :return: nothing, instead it updates the agents and their fitness.
        """

        # initializing the first set of agents
        self.agents = [self._mutate(self.initial_set, mutation_probability=1) for _ in range(self.num_agents)]
        self.agents = [self._project(agent) for agent in self.agents]

        if self.print_progress:
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
        if self.print_progress:
            print(f'Initialization done, took {end_time - start_time:.2f} seconds')
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f'Initialization done, took {end_time - start_time:.2f} seconds\n')
                for agent, fitness in zip(self.agents, self.fitness):
                    f.write(f'Agent: {agent}, Fitness: {fitness}\n')
                f.write('\n\n')

    def run_evolution(self):
        """
        Used to run the genetic algorithm.
        :return: nothing, instead it updates the best agent and the best fitness.
        """

        if self.print_progress:
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
            self.agents = [self._project(agent) for agent in self.agents]

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
            if self.flags['maximize']:
                self.last_best_agent = self.agents[np.argmax(self.fitness)]
                self.last_best_fitness = np.max(self.fitness)
                if self.best_fitness is None or self.last_best_fitness > self.best_fitness:
                    self.best_agent = self.last_best_agent
                    self.best_fitness = self.last_best_fitness
            else:
                self.last_best_agent = self.agents[np.argmin(self.fitness)]
                self.last_best_fitness = np.min(self.fitness)
                if self.best_fitness is None or self.last_best_fitness < self.best_fitness:
                    self.best_agent = self.last_best_agent
                    self.best_fitness = self.last_best_fitness

            if self.print_progress:
                print(f'Generation {i + 1}/{self.num_generations} done, best fitness: {self.last_best_fitness}, '
                      f'best agent: {self.last_best_agent}\n'
                      f'It took {end_time - start_time:.2f} seconds\n')
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(f'Generation {i + 1}/{self.num_generations} done, best fitness: {self.last_best_fitness}, '
                            f'best agent: {self.last_best_agent}\n'
                            f'It took {end_time - start_time:.2f} seconds\n')
                    for agent, fitness in zip(self.agents, self.fitness):
                        f.write(f'Agent: {agent}, Fitness: {fitness}\n')
                    f.write('\n\n')

    @staticmethod
    def _identity_projection(value):
        """
        Identity function used when no projection is needed.
        :param value: value to project.
        :return: that same value.
        """
        return value

    def _default_mutation(self, parameter_type, value):
        """
        Default mutation function used when no mutation is provided.
        :param parameter_type: type of the value that is being mutated.
        :param value: value to mutate.
        :return: mutated value.
        """
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
        """
        Inner function used to mutate the agent.
        :param agent: agent to mutate.
        :param mutation_probability: probability of mutation for each parameter.
        :return: mutated agent.
        """
        if mutation_probability is None:
            mutation_probability = self.mutation_probability
        # in order to avoid changing the original agent, we copy it
        new_agent = agent.copy()
        for key in agent.keys():
            if np.random.rand() < mutation_probability:
                new_agent[key] = self.parameter_mutations[key](self.parameter_types[key], agent[key])
        return new_agent

    def _crossover(self, agent1: dict, agent2: dict) -> (dict, dict):
        """
        Inner function used to crossover two agents.
        :param agent1: first agent.
        :param agent2: second agent.
        :return: two new agents.
        """
        new_agent1 = {}
        new_agent2 = {}

        # crossover happens with a certain probability
        if np.random.rand() < self.crossover_probability:
            # selecting the random crossover point
            crossover_point = np.random.randint(0, len(agent1))

            for i, key in self.parameter_order:
                # before the crossover point, the new agents are the same as the old ones
                # after the crossover point, the new agents are swapped
                if i < crossover_point:
                    new_agent1[key] = agent1[key]
                    new_agent2[key] = agent2[key]
                else:
                    new_agent1[key] = agent2[key]
                    new_agent2[key] = agent1[key]

        else:
            # if the crossover does not happen, the new agents are the same as the old ones
            new_agent1 = agent1
            new_agent2 = agent2

        return new_agent1, new_agent2

    def _scaler(self, values: list) -> list:
        """
        Inner function used to scale the fitness of the agents.
        Makes sure that the fitness is strictly positive, and flips the values in case the genetic algorithm is set
        to minimize the fitness.
        :param values: values to scale.
        :return: scaled values.
        """
        values = np.array(values)
        # flipping the values, if the genetic algorithm is set to minimize the fitness
        if not self.flags['maximize']:
            values = -values
        # finding the average value
        scale = (np.max(values)-np.mean(values))/len(values)
        if scale == 0:
            scale = 1
        # adding the minimal value and the scale to each value
        min_val = np.min(values)
        values = (values - min_val + scale)
        return values.tolist()

    def _select(self, agents: list, fitness: list) -> (dict, dict):
        """
        Inner function used to select two agents based on their fitness.
        :param agents: list of all agents.
        :param fitness: list of their fitness.
        :return: returns two selected agents.
        """
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
        """
        Helper function used to evaluate the fitness of the agents in parallel.
        :param agents: list of agents to evaluate.
        :param fitness: list to store fitness values.
        :param indices: indices of the agents to evaluate.
        :return: nothing, instead it updates the fitness list.
        """
        for i in indices:
            fitness[i] = self._evaluate(agents[i])

    def _evaluate(self, agent):
        """
        Used to evaluate the fitness of the agent.
        :param agent: agent to evaluate.
        :return: fitness of the agent.
        """
        return self.fitness_function(agent)

    def _project(self, agent):
        """
        Inner function used to project the agent to the desired range
        (by individually projecting each of the components).
        :param agent: agent to project.
        :return: projected agent.
        """
        projected_agent = {key: self.projections[key](value) for key, value in agent.items()}
        return projected_agent

    def _reproduce(self, agents, fitness):
        """
        Inner function used to reproduce the population of agents.
        :param agents: agents to reproduce.
        :param fitness: the fitness of the agents.
        :return: new population.
        """
        new_agents = []
        # all populations are of the same size
        while len(new_agents) < self.num_agents:
            # selecting two agents that would reproduce
            agent1, agent2 = self._select(agents, fitness)
            # creating two new agents by crossing over the selected agents
            new_agent1, new_agent2 = self._crossover(agent1, agent2)
            new_agents.append(new_agent1)
            new_agents.append(new_agent2)
        if len(new_agents) > self.num_agents:
            # if the number of new agents is odd, we need to remove one
            new_agents = new_agents[:self.num_agents]
        return new_agents

    def get_best_agent(self) -> (dict, float):
        """
        Returns the best agent produced by the genetic algorithm and its fitness.
        :return: tuple of the form (best_agent, best_fitness)
        """
        return (self.best_agent, self.best_fitness)

    @staticmethod
    def list_to_dict(keys: list, types: list = None, const_type = None):
        """
        If there are many keys with the same type, this function can be used to create a dictionary of the form
        {key: type} to be later passed to the GOE.
        :param keys: keys to create the dictionary from.
        :param types: list of types to assign to the keys. If None, uses the s_type parameter for all keys.
        :param const_type: in case all keys are of the same type, this parameter can be used to specify that type.
        :raises: ValueError if both types and s_type are None.
        :return: dictionary of the form {key: type}.
        """
        if not types and not const_type:
            raise ValueError('Either types or s_type should be provided')
        if not types:
            types = [const_type for _ in keys]
        return {key: type_ for key, type_ in zip(keys, types)}