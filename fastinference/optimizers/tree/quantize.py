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
    return np.array(leaf_probs).mean(axis=0).tolist()

def optimize(model, quantize_splits = None, quantize_leafs = None, **kwargs):
    """Quantizes the splits and predictions in the leaf nodes of the given tree and prunes away unreachable parts of the tree after quantization. 

    Note: The input data is **not** quantized as well if quantize_splits is set. Hence you have to manually scale the input data with the corresponding value to make sure splits are correctly performed.

    Args:
        model (Tree): The tree.
        quantize_splits (str or None, optional): Can be "rounding" or an integer (either int or as string). If "rounding" is set, then each split is rounded down towards the next integer. In any other case, quantize_splits is interpreted as integer value that is used to scale each split before rounding it down towards the next integer. Defaults to None.
        quantize_leafs (str or None, optional) : Can be a string or an integer. quantize_leafs is interpreted as integer value that is used to scale each leaf node before rounding it down towards the next integer. Defaults to None.
    Returns:
        Tree: The quantized and potentially pruned tree.
    """
    if quantize_splits is not None:
        for n in model.nodes:
            if n.split is not None:
                if quantize_splits == "rounding":
                    n.split = np.ceil(n.split).astype(int)
                else:
                    n.split = np.ceil(n.split * int(quantize_splits)).astype(int)
    
    # Prune the DT by removing sub-trees that are not accessible any more.
    fmin = [float('-inf') for _ in range(model.n_features)]
    fmax = [float('inf') for _ in range(model.n_features)]
    to_expand = [ (model.head, None, True, fmin, fmax) ]
    while len(to_expand) > 0:
        n, parent, is_left, fmin, fmax = to_expand.pop(0)
        if n.prediction is None:
            if not (fmin[n.feature] < n.split < fmax[n.feature]):
            #if fmax[n.fea]
            #if ((fmax[n.feature] is not None) and (fmax[n.feature] <= n.split)) or ((fmin[n.feature] is not None) and (fmin[n.feature] >= n.split)):
                new_node = Node()
                new_node.id = n.id
                new_node.numSamples = n.numSamples
                new_node.pathProb = n.pathProb
                new_node.prediction = get_leaf_probs(n)
                if parent is not None: 
                    if is_left: 
                        parent.leftChild = new_node
                    else:
                        parent.rightChild = new_node
                else:
                    print("WARNING: THIS SHOULD NOT HAPPEN IN optimizers.tree.quantize")
            else:
                lfmin, lfmax = fmin.copy(), fmax.copy()
                lfmax[n.feature] = n.split
                to_expand.append( (n.leftChild, n, True, lfmin, lfmax) )
            
                rfmin, rfmax = fmin.copy(), fmax.copy()
                rfmin[n.feature] = n.split
                to_expand.append( (n.rightChild, n, False, rfmin, rfmax) )

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

    if quantize_leafs is not None:
        for n in model.nodes:
            if n.prediction is not None:
                n.prediction = np.ceil(np.array(n.prediction) * int(quantize_leafs)).astype(int).tolist()

    return model