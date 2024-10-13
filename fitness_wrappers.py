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

def partial_fitness_wrapper(func: callable, **kwargs) -> callable:
    """
    Used to wrap a generic function to be used as a fitness function for the Genetic Optimization Engine.
    You can pass additional arguments to the function that are not parameters of the agents.

    :param func: function to evaluate the fitness of the agents. All of its parameters should be keyword arguments.
    :param kwargs: arguments to pass to the function that are not parameters of the agents.
    :return: function that receives a dictionary of the form {parameter_name: value} for each parameter.
    And calls the original function with the parameters as keyword arguments.
    """

    def fitness(parameters: dict) -> float:
        return func(**parameters, **kwargs)

    return fitness

def model_fitness_wrapper(model, func, func_arguments: dict):
    """
    Used to wrap the model for the GeneticParameterTuner
    :param model: model to evaluate the fitness of
    :param func: function to evaluate the fitness of the model
    :param func_arguments: arguments to the function. These arguments should not include the model,
    it is passed automatically
    :return: fitness of the model
    """

    def fitness(model_arguments: dict):
        return func(model=model.__class__(**model_arguments), **func_arguments)

    return fitness