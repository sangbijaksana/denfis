import numpy as np

class NormalizationLayer:
    """
    This class normalize the list by dividing it with sum of the list.
    """
    def __init__(self):
        pass

    def evaluate(self, old_evaluated_values):
        evaluated_values = []
        for i in old_evaluated_values:
            evaluated_values.append(i / np.sum(old_evaluated_values))
        return evaluated_values
