def section_projection(lower, upper):
    """
    Creates a projection function for a section between the lower and upper bounds.

    :param lower: lower bound. If None, there is no lower bound.
    :param upper: upper bound. If None, there is no upper bound.
    :return: function that projects a value to the section.
    """
    def projection(value):
        if lower is None:
            return min(value, upper)
        if upper is None:
            return max(value, lower)
        return max(min(value, upper), lower)
    return projection