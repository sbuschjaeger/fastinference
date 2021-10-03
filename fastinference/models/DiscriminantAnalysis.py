import json, os
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from .Linear import linear_from_sklearn, linear_from_json, linear_to_json_file, linear_to_json

from .Model import Model

class DiscriminantAnalysis(Model):
    """
    Placeholder class for all discriminant analysis models. Currently targetd towards scikit-learns QuadraticDiscriminantAnalysis.
    """
    def __init__(self, num_classes, accuracy = None, name = "Model"):
        """Constructor of this DiscriminantAnalysis.

        Args:
            num_classes (int): The number of classes this tree has been trained on.
            accuracy (float, optional): The accuracy of this tree on some test data. Can be used to verify the correctness of the implementation. Defaults to None.
            name (str, optional): The name of this model. Defaults to "Model".
        """
        super().__init__(num_classes, "discriminant", accuracy, name)
        self.product = []
        self.means = []
        self.log_priors = []
        self.rotations = []
        self.scalings = []
        self.scale_log_sums = []
    
    @classmethod
    def from_sklearn(cls, sk_model, name = "Model", accuracy = None):
        """Generates a new DiscriminantAnalysis from an sklearn QuadraticDiscriminantAnalysis.

        Args:
            sk_model (QuadraticDiscriminantAnalysis): A scikit-learn QuadraticDiscriminantAnalysis object. 
            name (str, optional): The name of this model. Defaults to "Model".
            accuracy (float, optional): The accuracy of this tree on some test data. Can be used to verify the correctness of the implementation. Defaults to None.

        Returns:
            DiscriminantAnalysis: The newly generated DiscriminantAnalysis object.
        """
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
        """Generates a new DiscriminantAnalysis from the given dictionary. It is assumed that the ensemble has previously been stored with the :meth:`DiscriminantAnalysis.to_dict` method.

        Args:
            data (dict): The dictionary from which this DiscriminantAnalysis should be generated. 

        Returns:
            DiscriminantAnalysis: The newly generated DiscriminantAnalysis classifier.
        """
        obj = DiscriminantAnalysis(data["num_classes"], data["category"], data.get("accuracy", None), data.get("name", "Model"))

        obj.means = data["means"] 
        obj.log_priors = data["log_priors"] 
        obj.rotations = data["rotations"] 
        obj.scalings = data["scalings"] 
        obj.product = data["product"] 
        obj.scale_log_sums = data["scale_log_sums"] 
        return obj

    def predict_proba(self,X):
        """Applies this DiscriminantAnalysis model to the given data and provides the predicted probabilities for each example in X.

        Args:
            X (numpy.array): A (N,d) matrix where N is the number of data points and d is the feature dimension. If X has only one dimension then a single example is assumed and X is reshaped via :code:`X = X.reshape(1,X.shape[0])`

        Returns:
            numpy.array: A (N, c) prediction matrix where N is the number of data points and c is the number of classes
        """
        if len(X.shape) == 1:
            X = X.reshape(1,X.shape[0])
        
        norm2 = []
        for i in range(len(self.classes_)):
            R = self.rotations[i]
            S = self.scalings[i]
            Xm = X - self.means[i]
            X2 = np.dot(Xm, R * (S ** (-0.5)))
            norm2.append(np.sum(X2 ** 2, axis=1))
        norm2 = np.array(norm2).T  # shape = [len(X), n_classes]
        u = np.asarray([np.sum(np.log(s)) for s in self.scalings])
        return (-0.5 * (norm2 + u) + np.log(self.log_priors))

    def to_dict(self):
        """Stores this DiscriminantAnalysis model as a dictionary which can be loaded with :meth:`DiscriminantAnalysis.from_dict`.

        Returns:
            dict: The dictionary representation of this DiscriminantAnalysis model.
        """
        model_dict = super().to_dict()
        model_dict["means"] = self.means
        model_dict["log_priors"] = self.log_priors
        model_dict["rotations"] = self.rotations
        model_dict["scalings"] = self.scalings
        model_dict["product"] = self.product
        model_dict["scale_log_sums"] = self.scale_log_sums

        return model_dict
    