class ProductLayer:
    """
    This class generate the product from each of adjacent membership function.
    """
    def __init__(self):
        self.membership_function_list = []

    def add_membership_function(self, membership_function):
        self.membership_function_list.append(membership_function)

    def evaluate(self, input_values):
        evaluated_membership_function = [mf.evaluate(input_values) for mf in self.membership_function_list]
        evaluated_values = []

        for i in range(len(evaluated_membership_function)-1):
            retrieved_val = evaluated_membership_function[i]*evaluated_membership_function[i+1]
            evaluated_values.append(retrieved_val)
        return evaluated_values
