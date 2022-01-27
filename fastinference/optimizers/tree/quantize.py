from hashlib import new
import numpy as np
from fastinference.models.Tree import Node

def get_leaf_probs(node):
    """Get the class probabilities of the subtree at the given node. 

    Args:
        node (Node): The root node of the subtree in question.

    Returns:
        np.array: Array of class probabilities
    """
    leaf_probs = []
    to_expand = [node]
    while len(to_expand) > 0:
        n = to_expand.pop(0)
        if n.prediction is not None:
            leaf_probs.append( np.array(n.prediction) ) # * n.numSamples ?
        else:
            to_expand.append(n.leftChild)
            to_expand.append(n.rightChild)
    return np.array(leaf_probs).mean(axis=0)

def optimize(model, quantize_splits = None, quantize_leafs = None, quantize_factor = 1000, **kwargs):
    """Quantizes the splits and predictions in the leaf nodes of the given tree and prunes away unreachable parts of the tree after quantization. 

    Note: If quantize_splits is set to "fixed" then input data is **not** quantized as well. Hence you have to manually scale the input data with quantize_factor to make sure splits are correctly applied.

    Args:
        model (Tree): The tree.
        quantize_splits (str or None, optional): Can be ["rounding", "fixed"]. If "rounding" is set, then each split is rounded down towards the next integer. If "fixed" is set, then each split is scaled by quantize_factor and then rounded down to the next integer. If any other string or None is given nothing happens. Defaults to None.
        quantize_leafs (str or None, optional): Can be ["fixed"]. If "fixed" is given then the probability estimates in the leaf nodes are scaled by quantize_factor and then rounded down. If any other string or None is given nothing happens. Defaults to None.
        quantize_factor (int, optional): The quantization_factor. Defaults to 1000.

    Returns:
        Tree: The quantized and potentially pruned tree.
    """
    if quantize_splits in ["rounding", "fixed"]:
        for n in model.nodes:
            if n.prediction is None:
                if quantize_splits == "rounding":
                    n.split = np.ceil(n.split).astype(int)
                else:
                    n.split = np.ceil(n.split * quantize_factor).astype(int)
    
    if quantize_leafs == "fixed":
        for n in model.nodes:
            if n.prediction is not None:
                n.prediction = np.ceil(np.array(n.prediction) * quantize_factor).astype(int)

    # Prune the DT by removing sub-trees that are not accessible any more.
    fmin = [None for _ in range(model.n_features)]
    fmax = [None for _ in range(model.n_features)]
    to_expand = [ (model.head, fmin, fmax) ]
    while len(to_expand) > 0:
        n, fmin, fmax = to_expand.pop(0)
        if n.prediction is None:
            lfmin, lfmax = fmin.copy(), fmax.copy()
            if lfmax[n.feature] == n.split:
                new_node = Node()
                new_node.id = n.leftChild.id
                new_node.numSamples = n.leftChild.numSamples
                new_node.prediction = get_leaf_probs(n.leftChild)
                n.leftChild = new_node
            else:
                lfmax[n.feature] = n.split
                to_expand.append( (n.leftChild, lfmin, lfmax) )

            rfmin, rfmax = fmin.copy(), fmax.copy()
            if rfmin[n.feature] is not None and rfmin[n.feature] > n.split:
                new_node = Node()
                new_node.id = n.rightChild.id
                new_node.numSamples = n.rightChild.numSamples
                new_node.prediction = get_leaf_probs(n.rightChild)
                n.rightChild = new_node
            else:
                rfmin[n.feature] = n.split
                to_expand.append( (n.rightChild, rfmin, rfmax) )

    # Make sure that node ids are correct after pruning and that model.nodes only
    # contains those nodes that remained in the tree
    new_nodes = []
    to_expand = [ model.head ]
    while len(to_expand) > 0:
        n = to_expand.pop(0)
        n.id = len(new_nodes)
        new_nodes.append(n)

        if n.prediction is None:
            to_expand.append(n.leftChild)
            to_expand.append(n.rightChild)

    model.nodes = new_nodes

    return model