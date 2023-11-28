from denfis.ecm import EvolvingClusteringMethod
from denfis.product_layer import ProductLayer
from denfis.consequent_layer import ConsequentLayer
from denfis.normalization_layer import NormalizationLayer
from denfis.membership_function import MembershipFunction

class DENFIS:
    """
    This class apply the whole DENFIS model.
    """
    def __init__(self, threshold_diameter=0.5, width_of_triangle=1.7):
        self.ecm = EvolvingClusteringMethod(threshold_diameter)
        self.threshold_diameter = threshold_diameter
        self.width_of_triangle = width_of_triangle

        self.product_layer = ProductLayer()
        self.normalization_layer = NormalizationLayer()
        self.consequent_layer = ConsequentLayer()

    # Train the model, generate the rule, get the weight of consequent layer
    def train(self, input_values, output_values):
        for input_value in zip(input_values):
            self.ecm.update_clusters(input_value)

        for cluster_center in self.ecm.cluster_centers:
            membership_function = MembershipFunction(cluster_center, self.threshold_diameter, self.width_of_triangle)
            self.product_layer.add_membership_function(membership_function)

        current_outputs = self.product_layer.evaluate(input_values)
        current_outputs = self.normalization_layer.evaluate(current_outputs)

        self.consequent_layer.train(current_outputs, input_values, output_values)

    # Predict new values after learning from training dataset
    def predict(self, new_input_values):
        predictions = self.product_layer.evaluate(new_input_values)
        predictions = self.normalization_layer.evaluate(predictions)
        predictions = self.consequent_layer.test(predictions, new_input_values)

        return predictions