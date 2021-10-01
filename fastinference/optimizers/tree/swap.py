from fastinference.models.Tree import Tree

def optimize(model, **kwargs):
    remaining_nodes = [model.head]

    while(len(remaining_nodes) > 0):
        cur_node = remaining_nodes.pop(0)

        if cur_node.probLeft >= cur_node.probRight:
            left = cur_node.leftChild
            right = cur_node.rightChild
            cur_node.leftChild = right
            cur_node.rightChild = left

    return model
