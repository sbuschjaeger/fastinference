import os
import json
from pathlib import Path

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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor

import xgboost as xgb

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
            "n_classes":model.n_classes,
            "classes":model.classes,
            "n_features":model.n_features,
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
        json.dump(data, outfile, indent=4, cls=NumpyEncoder)

    return os.path.join(out_path, file_name)

def model_from_dict(data):
    category = data.pop("category")
    if category == "linear":
        loaded_model = Linear.from_dict(data)
    elif category == "ensemble":
        loaded_model = fastinference.models.Ensemble.Ensemble.from_dict(data)
    elif category == "tree":
        loaded_model = Tree.from_dict(data)
    elif category == "discriminant":
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

        loaded_model = NeuralNet(file_path, name = data.get("name", "model"), accuracy = data.get("accuracy", None))
    else:
        if os.path.isfile( file_path ):
            with open(file_path) as f:
                data = json.load(f)

            loaded_model = model_from_dict(data)
    return loaded_model

def model_from_xgb_file(file_path):
    if os.path.isfile( file_path ) and (file_path.endswith("json") or file_path.endswith("JSON")):
        with open(file_path) as f:
            data = json.load(f)
            
            try:
                # parse to fit the needed format
                fi_model_dict = dict()
                fi_model_dict["category"] = "ensemble"
                num_classes = max(1, int(data["learner"]["learner_model_param"]["num_class"]))
                fi_model_dict["classes"] = list(range(num_classes))
                fi_model_dict["n_features"] = int(data["learner"]["learner_model_param"]["num_feature"])
                fi_model_dict["name"] = Path(file_path).stem
                # base_score is the "global bias" that needs to be added to the leaf predictions
                base_score = float(data["learner"]["learner_model_param"]["base_score"])

                # add trees to key "models"
                fi_model_dict["models"] = []
                
                xgb_trees = data["learner"]["gradient_booster"]["model"]["trees"]
                for xgb_tree in xgb_trees:

                    left_children = xgb_tree["left_children"]
                    right_children = xgb_tree["right_children"]
                    split_conditions = xgb_tree["split_conditions"]
                    split_indices = xgb_tree["split_indices"]

                    node_list = []
                    for i in range(len(split_conditions)):
                        node = dict()
                        node["pathProb"] = 0
                        node["numSamples"] = 0

                        if split_indices[i] != 0:
                            # is node
                            node["probLeft"] = 0
                            node["probRight"] = 0
                            node["isCategorical"] = False
                            node["feature"] = int(split_indices[i])
                            node["split"] = float(split_conditions[i])
                        else:
                            # is leaf
                            # add global bias to leaf prediction
                            node["prediction"] = [float(split_conditions[i]) + base_score]
                        
                        node_list.append(node)

                    # go through node_list and append node dicts to one another
                    for i in range(len(node_list)):
                        leftchild_index = left_children[i]
                        rightchild_index = right_children[i]

                        if leftchild_index != -1:
                            # has left child
                            node_list[i]["leftChild"] = node_list[leftchild_index]
                        if rightchild_index != -1:
                            # has right child
                            node_list[i]["rightChild"] = node_list[rightchild_index]
                        
                    # create tree dict
                    tree_dict = dict()
                    tree_dict["category"] = "tree"
                    tree_dict["classes"] = list(range(num_classes))
                    tree_dict["n_features"] = int(data["learner"]["learner_model_param"]["num_feature"])
                    # the first element of node_list is the root of the tree
                    tree_dict["model"] = node_list[0]

                    # create dict of weight and tree dictionary
                    tree_weight_dict = dict()
                    tree_weight_dict["weight"] = 1 / len(xgb_trees)
                    tree_weight_dict["model"] = tree_dict

                    fi_model_dict["models"].append(tree_weight_dict)

                return model_from_dict(fi_model_dict)
            except KeyError:
                print("""KeyError occured during parsing of json file:""")
                raise

    else:
        raise ValueError("""Received an unexpected file format. File format must be json.""")


def model_from_sklearn(sk_model, name, accuracy):
    """Loads a model from the given sklearn data structure. 
    Args:
        sk_model: A model previously fitted via scikit-learn.
        name (str): The name of the model. This name will later be used during code-generation for naming functions.
        accuracy (float): The accuracy of this model on some test-data. This can be used to verify the generated code.
    Raises:
        ValueError: If isinstance(sk_model, instances) is False where 
            instances = [RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier, LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, DecisionTreeRegressor, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor]
    Returns:
        Model: The loaded model.
    """    
    if isinstance(sk_model, (RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier)):
        return Linear.from_sklearn(sk_model, name, accuracy)
    elif isinstance(sk_model, (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)):
        return DiscriminantAnalysis.from_sklearn(sk_model, name, accuracy)
    elif isinstance(sk_model, (DecisionTreeRegressor, DecisionTreeClassifier)):
        return Tree.from_sklearn(sk_model, name, accuracy)
    elif isinstance(sk_model, (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor)):
        return fastinference.models.Ensemble.Ensemble.from_sklearn(sk_model, name, accuracy)
    else:
        raise ValueError("""
            Received and unrecognized sklearn model. 
            It was given: %s
            But currently supported are: 
            \tLINEAR: RidgeClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier,
            \tDISCRIMINANT: LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
            \tTREE: DecisionTreeRegressor, DecisionTreeClassifier
            \tENSEMBLE: RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        """ % type(sk_model).__name__)