import os
import json
import numpy as np

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor

import fastinference.Loader
from fastinference.models.Model import Model

class Ensemble(Model):
    """
    A Ensemble implementation. There is nothing fancy going on here. It stores all models of the ensemble in an array :code:`self.models` and the corresponding weights in :code:`self.weights`.
    """
    def __init__(self, num_classes, accuracy = None, name = "model"):
        """Constructor of this ensemble.

        Args:
            num_classes (int): The number of classes this tree has been trained on.
            accuracy (float, optional): The accuracy of this tree on some test data. Can be used to verify the correctness of the implementation. Defaults to None.
            name (str, optional): The name of this model. Defaults to "Model".
        """
        super().__init__(num_classes, "ensemble", accuracy, name)

        self.models = []
        self.weights = []
        
    @classmethod
    def from_sklearn(cls, sk_model, name = "model", accuracy = None):
        """Generates a new ensemble from an sklearn ensemble.

        Args:
            sk_model: A scikit-learn ensemble. Currently supported are :code:`{BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor}`
            name (str, optional): The name of this model. Defaults to "Model".
            accuracy (float, optional): The accuracy of this tree on some test data. Can be used to verify the correctness of the implementation. Defaults to None.

        Returns:
            Ensemble: The newly generated ensemble.
        """

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
        """Generates a new ensemble from the given dictionary. It is assumed that the ensemble has previously been stored with the :meth:`Ensemble.to_dict` method.

        Args:
            data (dict): The dictionary from which this ensemble should be generated. 

        Returns:
            Ensemble: The newly generated ensemble.
        """
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
        """Optimizes this ensemble and all of its base learners.

        Args:
            optimizers (list of str): A list of strings which should be used to optimize this ensemble.
            args (list of dict): A list of dictionaries containing the arguments for the respective optimizer.
            base_optimizers (list of str): A list of strings which should be used to optimize the base learners of this ensemble.
            base_args (list of dict): A list of dictionaries containing the arguments for the respective base optimizer.
        """
        super().optimize(optimizers, args)
        for e in self.models:
            e.optimize(base_optimizers, base_args)

    def implement(self, out_path, out_name, implementation_type, base_implementation, **kwargs):
        """Implements this ensemble.

        Args:
            out_path (str): Folder in which this ensemble should be stored.
            out_name (name): Filename in which this ensemble should be stored.
            implementation_type (str): The implementation which should be used to implement this ensemble, e.g. :code:`cpp`
            base_implementation (str): The implementation which should be used to implement the base learners, e.g. :code:`cpp.ifelse` for trees.
        """
        for m, w in zip(self.models, self.weights):
            m.implement(out_path = out_path, out_name = m.name, weight = w, implementation_type = base_implementation, **kwargs)

        super().implement(out_path, out_name, implementation_type, **kwargs)

    def predict_proba(self,X):
        """Applies this ensemble to the given data and provides the predicted probabilities for each example in X.

        Args:
            X (numpy.array): A (N,d) matrix where N is the number of data points and d is the feature dimension. If X has only one dimension then a single example is assumed and X is reshaped via :code:`X = X.reshape(1,X.shape[0])`

        Returns:
            numpy.array: A (N, c) prediction matrix where N is the number of data points and c is the number of classes
        """
        if len(X.shape) == 1:
            X = X.reshape(1,X.shape[0])
        
        all_proba = []
        for m, w in zip(self.models, self.weights):
            iproba = w * m.predict_proba(X)
            all_proba.append(iproba)
        all_proba = np.array(all_proba)

        return all_proba.sum(axis=0)

    def to_dict(self):
        """Stores this ensemble as a dictionary which can be loaded with :meth:`Ensemble.from_dict`.

        Returns:
            dict: The dictionary representation of this ensemble.
        """
        model_dict = super().to_dict()

        models = []
        for m,w in zip(self.models, self.weights):
            d = {}
            d["weight"] = w
            d["model"] = m.to_dict()
            models.append(d)
        model_dict["models"] = models

        return model_dict
