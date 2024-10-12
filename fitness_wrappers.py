
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
