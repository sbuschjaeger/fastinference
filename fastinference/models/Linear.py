import json, os
import numpy as np

import sklearn.linear_model

from sklearn.linear_model import RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from .Model import Model

class Linear(Model):
    """
    Placeholder class for all supported linear models.
    """    
    def __init__(self, num_classes, model_category, model_accuracy = None, model_name = "Model"):
        super().__init__(num_classes, model_category, model_accuracy, model_name)
        self.coef = []
        self.intercept = 0
    
    @classmethod
    def from_sklearn(cls, sk_model, name = "model", accuracy = None):
        model = Linear(len(set(sk_model.classes_)), "linear", accuracy, name)
        model.intercept = sk_model.intercept_
        model.coef = sk_model.coef_

        return model
    
    @classmethod
    def from_dict(cls, data):
        model = Linear(data["num_classes"], data["category"], data.get("accuracy", None), data.get("name", "Model"))
        model.intercept = data["intercept"]
        model.coef = data["coeff"]
        return model

    def to_dict(self):
        model_dict = super().to_dict()
        return {
            **model_dict,
            "intercept":self.intercept,
            "coeff":self.coef
        }
        