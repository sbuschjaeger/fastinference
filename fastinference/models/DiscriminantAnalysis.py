import json, os
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
# from .Linear import linear_from_sklearn, linear_from_json, linear_to_json_file, linear_to_json

from .Model import Model

class DiscriminantAnalysis(Model):
    """Placeholder class for all discriminant analysis models."""
    def __init__(self, num_classes, model_category, model_accuracy = None, model_name = "Model"):
        super().__init__(num_classes, model_category, model_accuracy, model_name)
        self.product = []
        self.means = []
        self.log_priors = []
        self.rotations = []
        self.scalings = []
        self.scale_log_sums = []
    
    @classmethod
    def from_sklearn(cls, sk_model, name = "Model", accuracy = None):
        obj = DiscriminantAnalysis(len(set(sk_model.classes_)), "discriminant", accuracy, name)

        obj.means = sk_model.means_
        obj.log_priors = np.log(sk_model.priors_)
        obj.rotations = sk_model.rotations_
        obj.scalings = sk_model.scalings_

        obj.product = np.asarray([r * (s ** (-0.5)) for (r,s) in zip(obj.rotations, obj.scalings)])
        obj.scale_log_sums = np.asarray([np.sum(np.log(s)) for s in obj.scalings])
        return obj

    @classmethod
    def from_dict(cls, data):
        obj = DiscriminantAnalysis(data["num_classes"], data["category"], data.get("accuracy", None), data.get("name", "Model"))

        obj.means = data["means"] 
        obj.log_priors = data["log_priors"] 
        obj.rotations = data["rotations"] 
        obj.scalings = data["scalings"] 
        obj.product = data["product"] 
        obj.scale_log_sums = data["scale_log_sums"] 
        return obj

    def to_dict(self):
        model_dict = super().to_dict()
        model_dict["means"] = self.means
        model_dict["log_priors"] = self.log_priors
        model_dict["rotations"] = self.rotations
        model_dict["scalings"] = self.scalings
        model_dict["product"] = self.product
        model_dict["scale_log_sums"] = self.scale_log_sums

        return model_dict
    