import os
import numpy as np
import heapq

from jinja2 import Environment, FileSystemLoader
import subprocess
    
def compute_node_sizes(tree, out_path, feature_type = "double", label_type = "double", weight = 1.0, compiler = "g++", objdump = "objdump"):
    """ Estimates the size of each node (in Bytes) in the tree by compiling dummy code with the given compiler and de-compiling it using objdump. Make sure that `compiler` and `objdump` are set correctly. If you want to use a cross-compiler (e.g. `arm-linux-gnueabihf-gcc`) you can set the path here accordingly. 

    Note 1: This code somewhat assume a gcc/g++ compiler because it uses `-std=c++11 -Wall -O3 -funroll-loops -ftree-vectorize` as compilation options.

    Reference:
        Chen, Kuan-Hsun et al. "Efficient Realization of Decision Trees for Real-Time Inference" 
        ACM Transactions on Embedded Computing Systems 2022

    Args:
        tree (Tree): The tree.
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        feature_type (str, optional): The data types of the input features. Defaults to "double".
        label_type (str, optional): The data types of the label. Defaults to "double".
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight. Defaults to 1.0.
        compiler (str, optional): The compiler used for compiling the dummy code to determine node sizes. If you want to use a cross-compiler (e.g. `arm-linux-gnueabihf-gcc`) you can set the path here accordingly. Defaults to "g++".
        objdump (str, optional): The de-compiler used for de-compiling the dummy code to determine node sizes. If you want to use a cross-compiler (e.g. `arm-linux-gnueabihf-gcc`) you can set the path here accordingly. Defaults to "objdump".

    Returns:
        list of int: A list of sizes in Bytes for each node in the given tree.
    """    

    #initialize node size table
    nodeSizeTable = []
    
    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
    )

    test_sections = env.get_template('testNodeSize.j2').render(
        tree = tree,
        feature_type = feature_type,
        label_type = label_type,
        weight = weight, 
    )

    with open(os.path.join(out_path, "{}.{}".format("sizeOfNode","cpp") ), 'w') as out_file:
        out_file.write(test_sections)

    subprocess.call(
        "{} {} -c -std=c++11 -Wall -O3 -funroll-loops -ftree-vectorize -o {}".format(compiler, os.path.join(out_path, "sizeOfNode.cpp"), os.path.join(out_path, "sizeOfNode.o")), shell=True
    )

    subprocess.call(
        "{} {} -h > {}".format(objdump, os.path.join(out_path, "sizeOfNode.o"), os.path.join(out_path, "sizeOfNode")), shell=True
    )

    #read sizeOfNode
    f = open(os.path.join(out_path, "sizeOfNode"), "r")
    lineList = f.readlines()
    leafE = 0
    splitE = 0 
    splitE1 = 0
    spReturnE = 0
    for i in range(len(lineList)):
        x = lineList[i].split()
        if(len(x) > 3):
            if(x[1][:4] == "test"):
                if x[1][5:9] == 'leaf':
                    nodeSizeTable.append([ int(x[1][10:]), int( x[2], 16) - leafE])
                elif x[1][5:12] == 'split__':
                    nodeSizeTable.append([ int(x[1][13:]), int( x[2], 16) - splitE1])
                else:
                    nodeSizeTable.append([ int(x[1][11:]), int( x[2], 16) - splitE])
            elif(x[1] == 'leafEmpty'):
                leafE = int( x[2], 16)
            elif(x[1] == 'splitEmpty'):
                splitE = int( x[2], 16) +spReturnE
            elif(x[1] == 'splitEmpty1'):
                splitE1 = int( x[2], 16) +spReturnE
            elif(x[1] == 'splitReturnEmpty'):
                spReturnE = int( x[2], 16) - leafE
    
    # Cleanup temp files. Should not be necessary since we are probably inside a temp folder anyway
    # try:
    #     os.remove(os.path.join(out_path, "sizeOfNode.cpp"))
    # except OSError:
    #     pass

    # try:
    #     os.remove(os.path.join(out_path, "sizeOfNode.o"))
    # except OSError:
    #     pass

    # try:
    #     os.remove(os.path.join(out_path, "sizeOfNode"))
    # except OSError:
    #     pass

    return [e[1] for e in nodeSizeTable]

def get_all_pathes(tree):
    """Returns all pathes from the root node the each leaf node in the given tree with the associated node probabilities.

    Args:
        tree (Tree): The tree.

    Returns:
        list of lists of tuples: Returns a list of lists of tuples: [ p1, p2, ...] where p1 = [(0, prob), (1, prob), (3, prob)....]
    """
    all_pathes = []
    to_expand = [ (tree.head, []) ]
    while len(to_expand) > 0:
        cur_node, path = to_expand.pop(0)
        
        if cur_node.prediction is not None:
            path.append((cur_node.id, 1))
            all_pathes.append(path)
        else:
            left_path = path.copy()
            left_path.append( (cur_node.id, cur_node.probLeft) )

            right_path = path.copy()
            right_path.append( (cur_node.id, cur_node.probRight) )

            to_expand.append( (cur_node.leftChild, left_path) )
            to_expand.append( (cur_node.rightChild, right_path) )

    return all_pathes

def path_sort(tree, budget, node_size):
    """Computes the nodes in the given tree which should be placed in the computation kernel given the budget by looking at the probability of entire pathes.

    Reference:
        Buschjäger, Sebastian, et al. "Realization of random forest for real-time evaluation through tree framing." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.

    Args:
        tree (Tree): The tree.
        budget (int): The allowed budget for the kernel in Bytes. Must be >= 0.
        target_architecture (str, optional): The target architecture which is used to determine the node size. Should be {arm, intel, ppc}. Defaults to "intel".

    Returns:
        dict: A dictionary which maps the node ids to True (=in Kernel) or False (=not in Kernel).
    """
    assert budget >= 0, "The budget must be >= 0."


    kernel = {}
    curSize = 0
    allPath = get_all_pathes(tree)

    paths = []
    for p in allPath:
        prob = 1
        path = []
        for (nid,nprob) in p:
            prob *= nprob
            path.append(nid)

        paths.append((path,prob))

    paths = sorted(paths, key=lambda x:x[1], reverse=True)

    for path in paths:
        for nodeid in path[0]:
            if not nodeid in kernel:
                if curSize >= budget:
                    kernel[nodeid] = False
                else:
                    curSize += node_size[nodeid]
                    kernel[nodeid] = True
    return kernel

def node_sort(tree, budget, node_size):
    """Computes the nodes in the given tree which should be placed in the computation kernel given the budget by looking at the probability of single nodes.

    Reference:
        Buschjäger, Sebastian, et al. "Realization of random forest for real-time evaluation through tree framing." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.

    Args:
        tree (Tree): The tree.
        budget (int): The allowed budget for the kernel in Bytes. Must be >= 0.
        target_architecture (str, optional): The target architecture which is used to determine the node size. Should be {arm, intel, ppc}. Defaults to "intel".

    Returns:
        dict: A dictionary which maps the node ids to True (=in Kernel) or False (=not in Kernel).
    """
    assert budget >= 0, "The budget must be >= 0."

    kernel = {}
    curSize = 0
    L = []
    heapq.heapify(L)
    nodes = [ tree.head ]
    while len(nodes) > 0:
        node = nodes.pop(0)
        
        if node.leftChild is not None:
            nodes.append( node.leftChild )

        if node.rightChild is not None:
            nodes.append( node.rightChild )
        # also add the node.id so that the sorting of the tuple is unique since heapq will evaluate
        # more tuple entries if the first one is the same
        heapq.heappush(L, (-node.pathProb, node.id, node))

    # now L has BFS nodes sorted by probabilities
    while len(L) > 0:
        _, _, node = heapq.heappop(L)
        curSize += node_size[node.id]
        # if the current size is larger than budget already, break.
        if curSize >= budget:
            kernel[node.id] = False
        else:
            kernel[node.id] = True

    return kernel

def to_implementation(model, out_path, out_name, weight = 1, namespace = "FAST_INFERENCE", feature_type = "double", label_type = "double", kernel_budget = None, kernel_type = None, output_debug = False, target_compiler = "g++", target_objdump = "objdump",**kwargs):
    """Generates a (unrolled) C++ implementation of the given Tree model. Unrolled means that the tree is represented in an if-then-else structure without any arrays. You can use this implementation by simply passing :code:`"cpp.ifelse"` to the implement, e.g.

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.json")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp.ifelse")

    Args:
        model (Tree): The Tree model to be implemented
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight. Defaults to 1.0.
        namespace (str, optional): The namespace under which this model will be generated. Defaults to "FAST_INFERENCE".
        feature_type (str, optional): The data types of the input features. Defaults to "double".
        label_type (str, optional): The data types of the label. Defaults to "double".
        quantize_splits (str, optional): Can be ["rounding", "fixed"] or None.
        kernel_budget (int, optional): The budget in bytes which is allowed in a single kernel. Kernel optimizations are ignored if the budget None. Defaults to None.
        kernel_type (str, optional): The type of kernel optimization. Can be {path, node, None}. Kernel optimizations are ignored if the kernel type is None. Defaults to None.
        output_debug (bool, optional): If True outputs the given tree in the given folder in a json file called `{model_name}_debug.json`. Useful when debugging optimizations or loading the tree with another tool. Defaults to False.
        target_compiler (str, optional): The compiler used for compiling the dummy code to determine node sizes. If you want to use a cross-compiler (e.g. `arm-linux-gnueabihf-gcc`) you can set the path here accordingly. Defaults to "g++".
        target_objdump (str, optional): The de-compiler used for de-compiling the dummy code to determine node sizes. If you want to use a cross-compiler (e.g. `arm-linux-gnueabihf-gcc`) you can set the path here accordingly. Defaults to "objdump".
    """

    if output_debug:
        from fastinference.Loader import model_to_json
        if out_path is None:
            out_path = "."
        model_to_json(model, out_path, "{}_debug.json".format(model.name))

    if kernel_budget is not None:
        assert kernel_type in ["node", "path"], "Only {node, path} kernels are currently supported."
        if kernel_type == "node":
            node_size = compute_node_sizes(model, out_path, feature_type, label_type, weight, target_compiler, target_objdump)
            kernel = node_sort(model, kernel_budget, node_size)
        elif kernel_type == "path":
            node_size = compute_node_sizes(model, out_path, feature_type, label_type, weight, target_compiler, target_objdump)
            kernel = path_sort(model, kernel_budget, node_size)
        else:
            kernel = None
    else:
        kernel = None

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
    )

    implementation = env.get_template('base.j2').render(
        model = model,
        model_name = model.name,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        code_static = "",
        weight = weight, 
        kernel = kernel
    )

    header = env.get_template('header.j2').render(
        model = model,
        model_name = model.name,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        model_weight = weight
    )

    # The Jinja2 whitespace handling is sometimes very weird. So far I was not able to 
    # generate code which looked the way I wanted it to look. Hence I sanitize the code
    # manually here. 
    # TODO Make this nicer and make sure that Labels: {} are correctly placed
    sanitized_impl = ""
    for line in implementation.split("\n"):
        if len(line) > 0 and not line.isspace():
            if line[0:4] == "    ":
                line = line[4:]
            sanitized_impl += line + "\n"

    with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
        out_file.write(sanitized_impl)

    with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
        out_file.write(header)
