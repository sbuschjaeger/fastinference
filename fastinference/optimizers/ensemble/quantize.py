from fastinference.models.Tree import Tree


def optimize(model, quantize_splits = None, quantize_leafs = None, **kwargs):

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

    for i in range(len(model.models)):
        h = model.models[i]
        w = model.weights[i]
        
        if isinstance(h, Tree):
            for n in h.nodes:
                if n.prediction is not None:
                    n.prediction = [p*w for p in n.prediction]

            h.optimize(["quantize"], {"quantize_splits":quantize_splits, "quantize_leafs":quantize_leafs})
            model.weights[i] = 1
    
    return model
            