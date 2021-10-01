import os
import json

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor

import fastinference.Loader
from fastinference.models.Model import Model

class Ensemble(Model):
    def __init__(self, num_classes, model_category, model_accuracy = None, model_name = "model"):
        super().__init__(num_classes, model_category, model_accuracy, model_name)

        self.models = []
        self.weights = []
        
    @classmethod
    def from_sklearn(cls, sk_model, name = "model", accuracy = None):
        model = Ensemble(len(set(sk_model.classes_)), "ensemble", accuracy, name)
         
        if isinstance(sk_model, (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor)):
            #obj.category = "ensemble"

            #if isinstance(sk_model, AdaBoostClassifier):
                #obj.type = "AdaBoostClassifier_" + sk_model.algorithm #AdaBoost Type SAMME, SAMME.R

            num_models = len(sk_model.estimators_)
            if isinstance(sk_model, (AdaBoostClassifier, AdaBoostRegressor)):
                model.weights = sk_model.estimator_weights_
            elif isinstance(sk_model, (GradientBoostingClassifier,GradientBoostingRegressor)):
                model.weights = [sk_model.learning_rate for _ in range(num_models*sk_model.n_classes_)] #weights are equal to the learning rate for GradientBoosting
                if sk_model.init_ != 'zero':
                    raise ValueError("""'zero' is the only supported init classifier for gradient boosting models""")
                    #TODO implement class prior classifier					 
            else:
                model.weights = [1.0/num_models for i in range(num_models)]

            model.models = []
            for i, base in enumerate(sk_model.estimators_):
                model.models.append(fastinference.Loader.model_from_sklearn(base, "{}_base_{}".format(name,i), accuracy))
        else:
            raise ValueError("""
                Received an unrecognized sklearn model. Expected was one of: 
                BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
            """ % type(sk_model).__name__)
        return model

    @classmethod
    def from_dict(cls, data):
        model = Ensemble(data["num_classes"], data["category"], data.get("accuracy", None), data.get("name", "Model"))
        #obj = super().from_dict(data)

        for entry in data["models"]:
            if "file" in entry:
                model.models.append(fastinference.Loader.model_from_file(entry["file"]))
            else:
                model.models.append(fastinference.Loader.model_from_dict(entry["model"]))
            model.weights.append(entry["weight"])

        return model

    def optimize(self, optimizers, args, base_optimizers, base_args):
        super().optimize(optimizers, args)
        for e in self.models:
            e.optimize(base_optimizers, base_args)

    def implement(self, out_path, out_name, implementation_type, base_implementation, **kwargs):
        for m, w in zip(self.models, self.weights):
            m.implement(out_path = out_path, out_name = m.name, weight = w, implementation_type = base_implementation, **kwargs)

        super().implement(out_path, out_name, implementation_type, **kwargs)

    def to_dict(self):
        model_dict = super().to_dict()

        models = []
        for m,w in zip(self.models, self.weights):
            d = {}
            d["weight"] = w
            d["model"] = m.to_dict()
            models.append(d)
        model_dict["models"] = models

        return model_dict
