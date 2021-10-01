import os
import json

import fastinference.models.Ensemble
from fastinference.models.Linear import Linear
from fastinference.models.Tree import Tree
from fastinference.models.DiscriminantAnalysis import DiscriminantAnalysis
#from fastinference.models.Ensemble import Ensemble
from fastinference.models.nn.NeuralNet import NeuralNet
from fastinference.Util import NumpyEncoder

from sklearn.linear_model import RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor

def model_to_json(model, out_path, file_name = "model"):
    """Saves the model to a json file. It requires to_dict to be implemented. 

    Args:
        model (obj): The model
        out_path (str): Path where the json file is stored.
        file_name (str): Filename under which the file is stored. The suffix `*.json` is automatically added if not present.
    """        

    assert not isinstance(model, NeuralNet), "Cannot store NeuralNet as JSON file"

    if not (file_name.endswith("json") or file_name.endswith("JSON")):
        file_name += ".json"

    if isinstance(model, fastinference.models.Ensemble.Ensemble):
        data = {
            "num_classes":model.num_classes,
            "category":model.category,
            "accuracy":model.accuracy,
            "name":model.name,
        }

        models = []
        for i,(m,w) in enumerate(zip(model.models, model.weights)):
            d = {}
            d["weight"] = w
            d["file"] = os.path.join(out_path, "{}_base_{}.json".format(model.name,i))
            models.append(d)
            model_to_json(m, out_path, "{}_base_{}".format(model.name,i))
        data["models"] = models
    else:
        data = model.to_dict()
            
    with open(os.path.join(out_path, file_name), "w") as outfile:  
        json.dump(data, outfile, cls=NumpyEncoder)

def model_from_dict(data):
    if data["category"] == "linear":
        loaded_model = Linear.from_dict(data)
    elif data["category"] == "ensemble":
        loaded_model = fastinference.models.Ensemble.Ensemble.from_dict(data)
    elif data["category"] == "tree":
        loaded_model = Tree.from_dict(data)
    elif data["category"] == "discriminant":
        loaded_model = DiscriminantAnalysis.from_dict(data)
        
    return loaded_model

def model_from_file(file_path):
    if file_path.endswith("onnx") or file_path.endswith("ONNX"):
        json_path = file_path.split(".onnx")[0] + ".json"
        if os.path.isfile( json_path ):
            with open(json_path) as f:
                data = json.load(f)
        else:
            data = {}

        loaded_model = NeuralNet(file_path, model_name = data.get("name", "model"), model_accuracy = data.get("accuracy", None))
    else:
        if os.path.isfile( file_path ):
            with open(file_path) as f:
                data = json.load(f)

            loaded_model = model_from_dict(data)
    return loaded_model

def model_from_sklearn(sk_model, name, accuracy):
    """Loads a model from the given sklearn data structure. 
    Args:
        sk_model: A model previously fitted via scikit-learn.
        name (str): The name of the model. This name will later be used during code-generation for naming functions.
        accuracy (float): The accuracy of this model on some test-data. This can be used to verify the generated code.
    Raises:
        ValueError: If isinstance(sk_model, instances) is False where 
            instances = [RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, DecisionTreeRegressor, DecisionTreeClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor]
    Returns:
        Model: The loaded model.
    """    
    if isinstance(sk_model, (RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier)):
        return Linear.from_sklearn(sk_model, name, accuracy)
    elif isinstance(sk_model, (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)):
        return DiscriminantAnalysis.from_sklearn(sk_model, name, accuracy)
    elif isinstance(sk_model, (DecisionTreeRegressor, DecisionTreeClassifier)):
        return Tree.from_sklearn(sk_model, name, accuracy)
    elif isinstance(sk_model, (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor)):
        return fastinference.models.Ensemble.Ensemble.from_sklearn(sk_model, name, accuracy)
    else:
        raise ValueError("""
            Received and unrecognized sklearn model. 
            It was given: %s
            But currently supported are: 
            \tLINEAR: RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier,
            \tDISCRIMINANT: LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
            \tTREE: DecisionTreeRegressor, DecisionTreeClassifier
            \tENSEMBLE: BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        """ % type(sk_model).__name__)