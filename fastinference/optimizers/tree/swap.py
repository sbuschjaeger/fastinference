from fastinference.models.Tree import Tree

def optimize(model, **kwargs):
    """Performs swap optimization. Swaps two child nodes if the probability to visit the left tree is smaller than the probability to visit the right tree. This way, the probability to visit the left tree is maximized which in-turn improves the branch-prediction during pipelining in the CPU. 
    You can activate this optimization by simply passing :code:`"swap"` to the optimizer, e.g.

    .. code-block::

        loaded_model = fastinference.Loader.model_from_file("/my/nice/tree.json")
        loaded_model.optimize("swap", None)

    Reference:
        Buschjäger, Sebastian, et al. "Realization of random forest for real-time evaluation through tree framing." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.

    Args:
        model (Tree): The tree model.

    Returns:
        Tree: The tree model with swapped nodes.
    """
    remaining_nodes = [model.head]

    while(len(remaining_nodes) > 0):
        cur_node = remaining_nodes.pop(0)

        if cur_node.probLeft < cur_node.probRight:
            left = cur_node.leftChild
            right = cur_node.rightChild
            cur_node.leftChild = right
            cur_node.rightChild = left

        if cur_node.prediction is not None:
            remaining_nodes.append(cur_node.leftChild)
            remaining_nodes.append(cur_node.rightChild)

    return model
