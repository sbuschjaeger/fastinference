# from abc import ABC, abstractmethod
import os
import json
from json import JSONEncoder
import numpy as np
import copy

from ..Util import NumpyEncoder, dynamic_import 
#XXX: This class currently also takes care of all ensemble related loading/storing. This is okayish, but should not become more complicated. Otherweise we should put this into the Ensemble directly. However, then storing models becomes weird because we have to provide the out_pathes etc to to_dict method. Also circular dependencies can become an issue then


class Model():
    """
    This class represents an abstract model. Its mainly used out of convienice to easily load / store models in various 
    formats, most notably to support sklearn models. Currently there is no real support to ship your own classifier 
    which means that you'll probably have to change some parts of this class as well.  
    In order to ship your own classifier you have to implement this class meaning you have to provide three functions:

    @classmethod
    def from_dict(cls, model_dict):
        ob = cls.__new__(cls)
        # Load your model from a dictionary

    @classmethod
    def from_sklearn(cls, sk_model):
        ob = cls.__new__(cls)
        # Load your model from a sklearn

    def to_dict(self):
        model_dict = {}
        # Write your model into a dictionary format

        return model_dict
    
    Note that `from_dict` and `from_sklearn` are factory functions used as constructors. 
    Its recommended that you provide `from_dict` and `to_dict` functions to easily debug your code. 
    However, if this does not make sense and/or your model was not trained via sklearn then feel free to throw an 
    appropriate error or return dummy values. In this case make sure that you set the following fields:
        - num_classes: The number of classes (=outputs of the model)
        - category: The category of the classifier. Feel free to add a new category if required, 
                    but make sure to change from_json_file accordingly
        - accuracy: Optional reference accuracy
        - model: The loaded model
    """
    def __init__(self, num_classes, model_category, model_accuracy = None, model_name = "Model"):
        """Generate a new Model.

        Args:
            num_classes (int): The number of classes (=outputs) predicted by the model 
            model_category (str): The category of the given model, e.g. `linear`, `tree`, `ensemble` or `neuralnet`. The category is used during code generation to load the appropriate templates.  
            model_accuracy (float, optional):  The accuracy of the given model (e.g. evaluated on some test data) which can be used to verify the correctness of the code generation. Defaults to None.
            model_name (str, optional): The name of the given model which is later used to name the functions during code generation.. Defaults to "Model".
        """
        # The number of classes and the model category must be set after reading from json / sklearn / onnx.
        # If this is not set, this is not a valid model! 
        # The category, e.g. `linear`, `tree`, `ensemble` or `neuralnet` is used during code generation to load the appropriate templates. 
        self.num_classes = num_classes
        self.category = model_category
        self.model = None

        # An optional reference accuracy for later checking of stuff
        self.accuracy = model_accuracy

        # The name of this model which is later used to generate the predict function, e.g. "RF_Large" leads to "predict_RF_Large"
        self.name = model_name

    def optimize(self, optimizers, args):
        if optimizers is None:
            return
            
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        if args is None:
            args = [{} for _ in optimizers]

        args = [{} if a is None else a for a in args]

        for opt, arg in zip(optimizers, args):
            run_optimization = dynamic_import("optimizers.{}.{}".format(self.category,opt), "optimize")
            self = run_optimization(self, **arg)

    def implement(self, out_path, out_name, implementation_type, **kwargs):
        to_implementation = dynamic_import("templates.{}.{}.implement".format(self.category,implementation_type), "to_implementation")
        self_copy = copy.deepcopy(self)
        to_implementation(self_copy, out_path, out_name, **kwargs)

    def to_dict(self):
        """Transforms this model into a dictionary format.

        Returns:
            dict: The dictionary representation of this model.
        """        
        model_dict = {}

        model_dict["num_classes"] = self.num_classes
        model_dict["category"] = self.category
        model_dict["accuracy"] = self.accuracy
        model_dict["name"] = self.name

        return model_dict
