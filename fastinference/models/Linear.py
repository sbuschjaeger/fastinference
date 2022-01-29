import json, os
import numpy as np

import sklearn.linear_model

from sklearn.linear_model import RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from .Model import Model

class Linear(Model):
    """
    A placeholder for all linear models. There is nothing fancy going on here. This class stores the coefficients of the linear function in :code:`self.coeff` and the bias/intercept in :code:`self.intercept`.
    """    
    def __init__(self, classes, n_features, accuracy = None, name = "Model"):
        """Constructor of a linear model.

        Args:
            classes (int): The class mappings. Each enty maps the given entry to the corresponding index so that the i-th output of the model belongs to class classes[i]. For example with classes = [1,0,2] the second output of the model maps to class 0, the first output to class 1 and the third output to class 2.
			n_features (list of int): The number of features this model was trained on.
            model_accuracy (float, optional): The accuracy of this tree on some test data. Can be used to verify the correctness of the implementation. Defaults to None.
            name (str, optional): The name of this model. Defaults to "Model".
        """
        super().__init__(classes, n_features, "linear", accuracy, name)
        self.coef = []
        self.intercept = []
    
    @classmethod
    def from_sklearn(cls, sk_model, name = "model", accuracy = None):
        """Generates a new linear model from sklearn. 

        Args:
            sk_model (LinearModel): A LinearModel trained in sklearn (e.g. SGDClassifier, RidgeClassifier, Perceptron, etc.).
            name (str, optional): The name of this model. Defaults to "Model".
            accuracy (float, optional): The accuracy of this tree on some test data. Can be used to verify the correctness of the implementation. Defaults to None.

        Returns:
            Linear: The newly generated linear model.
        """
        if len(sk_model.classes_) <= 2:
            model = Linear([0], sk_model.n_features_in_, accuracy, name)
        else:
            model = Linear(sk_model.classes_, sk_model.n_features_in_, accuracy, name)

        model.intercept = sk_model.intercept_
        model.coef = sk_model.coef_.T

        return model
    
    @classmethod
    def from_dict(cls, data):
        """Generates a new linear model from the given dictionary. It is assumed that a linear model has previously been stored with the :meth:`Linear.to_dict` method.

        Args:
            data (dict): The dictionary from which this linear model should be generated. 

        Returns:
            Tree: The newly generated linear model.
        """
        model = Linear(data["classes"], data["n_features"], data.get("accuracy", None), data.get("name", "Model"))
        model.intercept = np.array(data["intercept"])
        model.coef = np.array(data["coeff"])

        return model

    def predict_proba(self,X):
        """Applies this linear model to the given data and provides the predicted probabilities for each example in X.

        Args:
            X (numpy.array): A (N,d) matrix where N is the number of data points and d is the feature dimension. If X has only one dimension then a single example is assumed and X is reshaped via :code:`X = X.reshape(1,X.shape[0])`

        Returns:
            numpy.array: A (N, c) prediction matrix where N is the number of data points and c is the number of classes
        """
        if len(X.shape) == 1:
            X = X.reshape(1,X.shape[0])
        
        # Somewhat stolen and modified from safe_sparse_dot in sklearn extmath.py
        if X.ndim > 2 or self.coef.ndim > 2:
            proba = np.dot(X, self.coef)
        else:
            proba = X @ self.coef

        # proba = []
        # for x in X:
        #     proba.append(np.inner(x, self.coef) + self.intercept)
        return np.array(proba) + self.intercept
        
    def to_dict(self):
        """Stores this linear model as a dictionary which can be loaded with :meth:`Linear.from_dict`.

        Returns:
            dict: The dictionary representation of this linear model.
        """
        model_dict = super().to_dict()
        return {
            **model_dict,
            "intercept":self.intercept,
            "coeff":self.coef
        }
        