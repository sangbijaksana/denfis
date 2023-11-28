import numpy as np

class ConsequentLayer:
    """
    This class apply LSE to generate the least differece between predicted output and target output.
    """
    def __init__(self):
        self.consequent_parameter = None

    def train(self, weights, input_values, target_values):
        # Apply the LSE to solve for the rule consequents
        # Here we're solving X * B = Y, where B is the matrix of parameters we're trying to find
        # We use the pseudoinverse to find the best solution in a least squares sense

        function_weight = []
        for input_value in input_values:
            current_function_weight = []

            for i in input_value:
                current_function_weight.append(sum(weights)*i)
            current_function_weight.append(sum(weights))
            function_weight.append(current_function_weight)

        target_values = np.array(target_values).reshape(-1, 1)
        B, residuals, rank, s = np.linalg.lstsq(function_weight, target_values, rcond=None)
        self.consequent_parameter = B

    def test(self, weights, input_values):
        function_weights = []
        for input_value in input_values:
            current_function_weight = []

            for i in input_value:
                current_function_weight.append(sum(weights)*i)
            current_function_weight.append(sum(weights))
            function_weights.append(current_function_weight)

        predictions = np.matmul(function_weights, self.consequent_parameter)
        return predictions
