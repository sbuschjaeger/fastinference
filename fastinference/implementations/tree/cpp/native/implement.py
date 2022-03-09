import os
import numpy as np
import heapq

from jinja2 import Environment, FileSystemLoader

def reorder(model, set_size = 8, force_cacheline = False, **kwargs):
    """Extracts a list of inner_nodes and leaf_nodes from the model while storing additional left_is_leaf / right_is_leaf / id fields in the inner_nodes for the code generation. The left_is_leaf/right_is_leaf fields indicate if the left/right child of an inner node is a leaf note, whereas the id field can be used to access the correct index in the array, e.g. by using node.leftChild.id. This method tries to place nodes in consecutive order which have a maximum probability to be executed together. This basically implements algorithm 2 from the given reference.

    Reference:
        Buschjäger, Sebastian, et al. "Realization of random forest for real-time evaluation through tree framing." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.

    Args:
        model (Tree): A given tree model.
        set_size (int, optional): The cache set size. Defaults to 8.
        force_cacheline (bool, optional): If True then "padding" nodes are introduced to fill the entire cache line. Defaults to False.

    Returns:
        inner_nodes: The list of inner_nodes in the order given by the BFS
        leaf_nodes: The list of leaf_nodes in the order given by the BFS
    """
    leaf_nodes = []
    inner_nodes = []
    to_expand = []
    # Per convention heappush uses the first element of a tuple for comparisons. We are using
    #   (pathProb, parent id, true/false if this node is a left subtree, node)
    # to manage the nodes. Note that heapq maintains a min-heap, whereas we require a max-heap. 
    # Hence we use the negative pathProb.
    heapq.heappush(to_expand, (-model.head.pathProb,-1, False, model.nodes[0]))

    while( len(to_expand) > 0 ):
        # Extract the current node with its meta information. The pathProb can be ignored 
        _, pid, is_left, n = heapq.heappop(to_expand) 
        cset_size = 0

        # Is the current set full already? 
        while (cset_size < set_size):
            if n.prediction is not None:
                # A leaf node is found and hence this path cannot be explored further.
                if pid >= 0:

                    # Make sure the id of our parent node points to the correct index and set is_leaf correctly
                    if is_left:
                        inner_nodes[pid].leftChild.id = len(leaf_nodes)
                        inner_nodes[pid].left_is_leaf = "true"
                    else:
                        inner_nodes[pid].rightChild.id = len(leaf_nodes)
                        inner_nodes[pid].right_is_leaf = "true"

                if force_cacheline:
                    # Fill in padding / dummy nodes if cset is not full yet
                    for _ in range(cset_size - set_size):
                        inner_nodes.append(model.head)
                        cset_size += 1
                
                leaf_nodes.append(n)
                break
            else:
                # An inner node is found and hence we may explore this path. This node is added to the inner_nodes 
                # and hence cset_size increases by one
                cset_size += 1
                cid = len(inner_nodes)
                inner_nodes.append(n)

                # Just set the is_leaf fields to false for all nodes. If we reach an actual leaf node then we will
                # set it to "true" in the above code path
                n.left_is_leaf = "false"
                n.right_is_leaf = "false"
                
                if pid >= 0:
                    # Make sure the id of our parent node points to the correct index
                    if is_left:
                        inner_nodes[pid].leftChild.id = cid
                    else:
                        inner_nodes[pid].rightChild.id = cid

                # Directly explore the left / right sub-tree without using the heap. 
                # Put the other sub-tree on the heap for later. 
                # Since heappush maintains the heap-invariant there is not need to call heapify
                if cset_size < set_size:
                    if n.leftChild.pathProb > n.rightChild.pathProb:
                        heapq.heappush(to_expand, (-n.rightChild.pathProb, cid, False, n.rightChild))
                        pid, is_left, n = cid, True, n.leftChild
                    else:
                        heapq.heappush(to_expand, (-n.leftChild.pathProb, cid, True, n.leftChild))
                        pid, is_left, n = cid, False, n.rightChild
                else:
                    # If the set size is already full then continue normally by including both children into the heap
                    heapq.heappush(to_expand, (-n.rightChild.pathProb, cid, False, n.rightChild))
                    heapq.heappush(to_expand, (-n.leftChild.pathProb, cid, True, n.leftChild))

    return inner_nodes, leaf_nodes

def get_nodes(model):
    """Extracts a list of inner_nodes and leaf_nodes from the model while storing additional left_is_leaf / right_is_leaf / id fields in the inner_nodes for the code generation. The left_is_leaf/right_is_leaf fields indicate if the left/right child of an inner node is a leaf note, whereas the id field can be used to access the correct index in the array, e.g. by using node.leftChild.id. This method traverses the tree in BFS order and does not perform any optimizations on the order.

    Args:
        model (Tree): A given tree model.

    Returns:
        inner_nodes: The list of inner_nodes in the order given by the BFS
        leaf_nodes: The list of leaf_nodes in the order given by the BFS
    """
    leaf_nodes = []
    inner_nodes = []
    to_expand = [(-1, False, model.nodes[0]) ]

    # Make sure that the nodes are correctly numbered given their current order
    # To do so, traverse the tree in BFS order and maintain a tuple (parent id, true/false if this is a left child, node)
    # We also split the inner nodes and the leaf nodes into two arrays inner_nodes and leaf_nodes
    # Last we make sure to set the left_is_leaf / right_is_leaf fields of the node which is then accessed during code generation
    while( len(to_expand) > 0 ):
        pid, is_left, n = to_expand.pop(0)

        if n.prediction is not None:
            if pid >= 0:
                # Make sure the id of our parent node points to the correct index and set is_leaf correctly
                if is_left:
                    inner_nodes[pid].leftChild.id = len(leaf_nodes)
                    inner_nodes[pid].left_is_leaf = "true"
                else:
                    inner_nodes[pid].rightChild.id = len(leaf_nodes)
                    inner_nodes[pid].right_is_leaf = "true"

            leaf_nodes.append(n)
        else:
            cid = len(inner_nodes)
            inner_nodes.append(n)

            # Just set the is_leaf fields to false for all nodes. If we reach an actual leaf node then we will
            # set it to "true" in the above code path
            n.left_is_leaf = "false"
            n.right_is_leaf = "false"
            
            if pid >= 0:
                # Make sure the id of our parent node points to the correct index
                if is_left:
                    inner_nodes[pid].leftChild.id = cid
                else:
                    inner_nodes[pid].rightChild.id = cid

            to_expand.append( (cid, True, n.leftChild) )
            to_expand.append( (cid, False, n.rightChild) )
    return inner_nodes, leaf_nodes

def to_implementation(model, out_path, out_name, weight = 1, namespace = "FAST_INFERENCE", feature_type = "double", label_type = "double", int_type = "unsigned int", output_debug = False, infer_types = True, reorder_nodes = False, set_size = 8, force_cacheline = False, **kwargs):
    """Generates a native C++ implementation of the given Tree model. Native means that the tree is represented in an array structure which is iterated via a while-loop. You can use this implementation by simply passing :code:`"cpp.native"` to the implement, e.g.

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.json")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp.native")

    Args:
        model (Tree): The Tree model to be implemented
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight. Defaults to 1.0.
        namespace (str, optional): The namespace under which this model will be generated. Defaults to "FAST_INFERENCE".
        feature_type (str, optional): The data types of the input features. Defaults to "double".
        label_type (str, optional): The data types of the label. Defaults to "double".
        output_debug (bool, optional): If True outputs the given tree in the given folder in a json file called `{model_name}_debug.json`. Useful when debugging optimizations or loading the tree with another tool. Defaults to False.
        infer_types (bool, optional): If True then the smallest data type for index variables is inferred by the overall tree size. Otherwise "unsigned int" is used. Defaults to False.
        reorder_nodes (bool, optional): If True then the nodes in the tree are reorder so that cache set size is respected. You can set the size of the cache set via set_size parameter. Defaults to False.
        set_size (int, optional): The size of the cache set for if reorder_nodes is set to True. Defaults to 8.
        force_cacheline (bool, optional): If True then "padding" nodes are introduced to fill the entire cache line. Defaults to False.
    """

    if output_debug:
        from fastinference.Loader import model_to_json
        if out_path is None:
            out_path = "."
        model_to_json(model, out_path, "{}_debug.json".format(model.name))

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    if reorder_nodes:
        inner_nodes, leaf_nodes = reorder(model, set_size, force_cacheline)
    else:
        inner_nodes, leaf_nodes = get_nodes(model)
    
    if infer_types:
        if len(inner_nodes) < 2**8:
            int_type = "unsigned char"
        elif len(inner_nodes) < 2**16:
            int_type = "unsigned short"
        elif len(inner_nodes) < 2**32:
            int_type = "unsigned int"
        else:
            int_type = "unsigned long"

        # It might be interstring to use a different split_type for splits, but this can get weird if split_type != feautre_type
        # So lets leave this code here for now and see what happens
        
        # split_type = None
        # min_split = None
        # max_split = None
        # for n in inner_nodes:
        #     if isinstance(n.split, (np.float, float)):
        #         split_type = "double"
        #         break
        #     else:
        #         if max_split is None or n.split > max_split:
        #             max_split = n.split
        #         if min_split is None or n.split < min_split:
        #             min_split = n.split
        
        # if split_type is None:
        #     if min_split is None or max_split is None:
        #         split_type = "double"
        #     elif max_split < 2**8 and min_split > 0:
        #         split_type = "unsigned char"
        #     elif max_split < 2**7 and min_split > -2**7:
        #         split_type = "char"
        #     elif max_split < 2**16 and min_split > 0:
        #         split_type = "unsigned short"
        #     elif max_split < 2**15 and min_split > -2**15:
        #         split_type = "short"
        #     elif max_split < 2**32 and min_split > 0:
        #         split_type = "unsigned int"
        #     elif max_split < 2**31 and min_split > -2**31:
        #         split_type = "int"
        #     elif min_split > 0:
        #         split_type = "unsigned long"
        #     else:
        #         split_type = "long"

    implementation = env.get_template('base.j2').render(
        model = model,
        model_name = model.name,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        code_static = "",
        weight = weight,
        int_type = int_type,
        leaf_nodes = leaf_nodes,
        inner_nodes = inner_nodes,
    )

    header = env.get_template('header.j2').render(
        model = model,
        model_name = model.name,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        model_weight = weight,
        int_type = int_type,
        leaf_nodes = leaf_nodes,
        inner_nodes = inner_nodes,
    )

    with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
        out_file.write(implementation)

    with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
        out_file.write(header)