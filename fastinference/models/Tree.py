import json, os
import numpy as np

from sklearn.tree import _tree
from sklearn.tree import DecisionTreeRegressor		

from .Model import Model

class Node:
    def __init__(self):
        # The ID of this node. Makes addressing sometimes easier 
        self.id = None
        
        # The total number of samples seen at this node 
        self.numSamples = None
        
        # The probability to follow the left child
        self.probLeft = None
        
        # The probability to follow the right child
        self.probRight = None

        # An array of predictions, where each entry represents the class weight / probability
        # The specific meaning of each entry depends on the specific method (AdaBoost vs RandomForest)
        self.prediction = None

        # The 'way' of comparison. Usually this is "<=". However, for non-numeric 
        # features, one sometimes need equals comparisons "==". This is currently not 
        # implemented. 
        self.isCategorical = None

        # The index of the feature to compare against
        self.feature = None

        # The threashold the features gets compared against
        self.split = None

        # The right child of this node inside the tree
        self.rightChild = None

        # The left child of this node inside the tree
        self.leftChild = None

        # The probability of this node accumulated from the probabilities of previous
        # edges on the same path.
        # Note: This field is only used after calling getProbAllPaths onc
        self.pathProb = None

class Tree(Model):
    def __init__(self, num_classes, model_category, model_accuracy = None, model_name = "Model"):
        super().__init__(num_classes, model_category, model_accuracy, model_name)

        # Array of all nodes
        self.nodes = []

        # Pointer to the root node of this tree
        self.head = None

    @classmethod
    def from_sklearn(cls, sk_model, name = "Model", accuracy = None, ensemble_type = None):
        tree = Tree(len(set(sk_model.classes_)), model_category = "tree", model_name = name, model_accuracy=accuracy)
        
        tree.nodes = []
        tree.head = None
        tree.category = "tree"

        sk_tree = sk_model.tree_

        node_ids = [0]
        tmp_nodes = [Node()]
        while(len(node_ids) > 0):
            cur_node = node_ids.pop(0)
            node = tmp_nodes.pop(0)

            node.numSamples = int(sk_tree.n_node_samples[cur_node])
            if sk_tree.children_left[cur_node] == _tree.TREE_LEAF and sk_tree.children_right[cur_node] == _tree.TREE_LEAF:
                # Get array of prediction probabilities for each class
                proba = sk_tree.value[cur_node][0, :]  
                #computations of prediction values for bagging and boosting models:
                if ensemble_type is None or ensemble_type == "RandomForestClassifier":
                    proba = proba / sum(proba)
                elif ensemble_type == "AdaBoostClassifier_SAMME.R":
                    proba = proba/sum(proba)
                    np.clip(proba, np.finfo(proba.dtype).eps, None, out = proba)
                    log_proba = np.log(proba)
                    proba = (log_proba - (1. /sk_model.n_classes_) * log_proba.sum())
                elif ensemble_type == "AdaBoostClassifier_SAMME":
                    proba = proba/sum(proba)
                    proba = proba #*weights
                elif isinstance(sk_model, DecisionTreeRegressor):
                    proba = proba #*weights

                node.prediction = proba #* weight
            else:
                node.feature = sk_tree.feature[cur_node]
                node.split = sk_tree.threshold[cur_node]

                node.isCategorical = False # Note: So far, sklearn does only support numrical features, see https://github.com/scikit-learn/scikit-learn/pull/4899
                samplesLeft = float(sk_tree.n_node_samples[sk_tree.children_left[cur_node]])
                samplesRight = float(sk_tree.n_node_samples[sk_tree.children_right[cur_node]])
                node.probLeft = samplesLeft / node.numSamples
                node.probRight = samplesRight / node.numSamples
            
            node.id = len(tree.nodes)
            if node.id == 0:
                tree.head = node

            tree.nodes.append(node)

            if node.prediction is None:
                leftChild = sk_tree.children_left[cur_node]
                node_ids.append(leftChild)
                node.leftChild = Node()
                tmp_nodes.append(node.leftChild)

                rightChild = sk_tree.children_right[cur_node]
                node_ids.append(rightChild)
                node.rightChild = Node()
                tmp_nodes.append(node.rightChild)

        return tree

    @classmethod
    def from_dict(cls, data):
        tree = Tree(data["num_classes"], data["category"], data.get("accuracy", None), data.get("name", "Model"))
        tree.nodes = []

        nodes = [Node()]
        tree.head = nodes[0]
        dicts = [data["model"]]
        while(len(nodes) > 0):
            node = nodes.pop(0)
            entry = dicts.pop(0)
            node.id = entry["id"]
            node.numSamples = int(entry["numSamples"])

            if "prediction" in entry and entry["prediction"] is not None:
                node.prediction = entry["prediction"]
            else:
                node.probLeft = float(entry["probLeft"])
                node.probRight = float(entry["probRight"])
                node.isCategorical = (entry["isCategorical"] == "True")
                node.feature = int(entry["feature"])
                node.split = entry["split"]
                node.rightChild = entry["rightChild"]["id"]
                node.leftChild = entry["leftChild"]["id"]
                node.pathProb = entry["pathProb"]

                tree.nodes.append(node)
                if node.prediction is None:
                    node.rightChild = Node()
                    nodes.append(node.rightChild)
                    dicts.append(entry["rightChild"])

                    node.leftChild = Node()
                    nodes.append(node.leftChild)
                    dicts.append(entry["leftChild"])
        return tree

    def _to_dict(self, node):
        model_dict = {}
        
        if node is None:
            node = self.head
        
        model_dict["id"] = node.id
        model_dict["probLeft"] = node.probLeft
        model_dict["probRight"] = node.probRight
        model_dict["prediction"] = node.prediction
        model_dict["isCategorical"] = node.isCategorical
        model_dict["feature"] = node.feature
        model_dict["split"] = node.split
        model_dict["pathProb"] = node.pathProb
        model_dict["numSamples"] = node.numSamples

        if node.rightChild is not None:
            model_dict["rightChild"] = self._to_dict(node.rightChild)

        if node.leftChild is not None:
            model_dict["leftChild"] = self._to_dict(node.leftChild)

        return model_dict

    def to_dict(self):
        model_dict = super().to_dict()
        model_dict["model"] = self._to_dict(self.head)
        return model_dict

    ## SOME STATISTICS FUNCTIONS ##

    # def getMaxDepth(self):
    #     paths = self.getAllPaths()
    #     return max([len(p) for p in paths])

    # def getAvgDepth(self):
    #     paths = self.getAllPaths()
    #     return sum([len(p) for p in paths]) / len(paths)

    # def getNumNodes(self):
    #     return len(self.nodes)
