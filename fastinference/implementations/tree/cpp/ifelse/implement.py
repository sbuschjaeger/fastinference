import os
import numpy as np
import heapq

from jinja2 import Environment, FileSystemLoader

def contains_float(tree):
    """Returns True if the given tree contains any floating point splits.

    Args:
        tree (Tree): The tree.

    Returns:
        bool: True if any split in tree is floating point, else False
    """
    for node in tree.nodes:
        if (isinstance(node.split, (np.float16, np.float32, np.float64, float))):
            return True

    return False

def node_size(node, tree, target_architecture = "intel"):
    """Heuristically determines the node size of the given node on the given architecture.
    TODO: This MUST be refactored. All estimations are currently most likely wrong. 

    Args:
        node (Node): The node whose size should be estimated. 
        tree (Tree): The tree which contains the given node.
        target_architecture (str, optional): The target architecture which is used to determine the node size. Should be {arm, intel, ppc}. Defaults to "intel".

    Returns:
        int: The estimated size of the node in Bytes.
    """
    assert target_architecture in ["intel", "arm", "ppc"], "The target architecture must be {intel, arm, ppc}."

    has_float = contains_float(tree)
    size = 0

    if node.prediction is not None:
        if not has_float and target_architecture == "arm":
            size += 2*4
        elif has_float and target_architecture == "arm":
            size += 2*4
        elif not has_float and target_architecture == "intel":
            size += 10
        elif has_float and target_architecture == "intel":
            size += 10
        elif not has_float and target_architecture == "ppc":
            size += 2*4
        elif has_float and target_architecture == "ppc":
            size += 2*4
    else:
        # In O0, the basic size of a split node is 4 instructions for loading.
        # Since a split node must contain a pair of if-else statements,
        # one instruction for branching is not avoidable.
        if not has_float and target_architecture == "arm":
            # this is for arm int (ins * bytes)
            size += 5*4
        elif has_float and target_architecture == "arm":
            # this is for arm float
            size += 8*4
        elif not has_float and target_architecture == "ppc":
            # this is for ppc int (ins * bytes)
            size += 5*4
        elif has_float and target_architecture == "ppc":
            # this is for ppc float
            size += 8*4
        elif not has_float and target_architecture == "intel":
            # this is for intel integer (bytes)
            size += 28
        elif has_float and target_architecture == "intel":
            # this is for intel float (bytes)
            size += 17
    return size

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

def path_sort(tree, budget, target_architecture = "intel"):
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
    assert target_architecture in ["intel", "arm", "ppc"], "The target architecture must be {intel, arm, ppc}."
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
                    curSize += searchNodeSizeTable(tree, nodeid)
                    kernel[nodeid] = True
    return kernel

def node_sort(tree, budget, target_architecture = "intel"):
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
    assert target_architecture in ["intel", "arm", "ppc"], "The target architecture must be {intel, arm, ppc}."
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
        curSize += searchNodeSizeTable(tree, node.id)
        # if the current size is larger than budget already, break.
        if curSize >= budget:
            kernel[node.id] = False
        else:
            kernel[node.id] = True

    return kernel

def to_implementation(model, out_path, out_name, weight = 1.0, namespace = "FAST_INFERENCE", feature_type = "double", label_type = "double", round_splits = False, kernel_budget = None, kernel_type = None, target_architecture = "intel", **kwargs):
    """Generates a (unrolled) C++ implementation of the given Tree model. Unrolled means that the tree is represented in an if-then-else structure without any arrays. You can use this implementation by simply passing :code:`"cpp.ifelse"` to the implement, e.g.

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.json")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp.ifelse")

    Args:
        model (Tree): The Tree model to be implemented
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight.. Defaults to 1.0.
        namespace (str, optional): The namespace under which this model will be generated. Defaults to "FAST_INFERENCE".
        feature_type (str, optional): The data types of the input features. Defaults to "double".
        label_type (str, optional): The data types of the label. Defaults to "double".
        round_splits (bool, optional): If True then all splits are rounded towards the next integer. Defaults to False.
        kernel_budget (int, optional): The budget in bytes which is allowed in a single kernel. Kernel optimizations are ignored if the budget None. Defaults to None.
        kernel_type (str, optional): The type of kernel optimization. Can be {path, node, None}. Kernel optimizations are ignored if the kernel type is None. Defaults to None.
        target_architecture (str, optional): The target architecture {intel, arm, ppc} which is used to estimate the node size for kernel optimizations. Defaults to "intel".
    """

    # Give testing data
    kernel_budget = 100
    kernel_type = "path"

    if round_splits:
        for n in model.nodes:
            if n.prediction is None:
                n.split = np.ceil(n.split).astype(int)

    if kernel_budget is not None:
        assert target_architecture in ["intel", "arm", "ppc"], "Only {intel, arm, ppc} are currently supported to estimate the size tree nodes."
        assert kernel_type in ["node", "path"], "Only {node, path} kernels are currently supported."
        if kernel_type == "node":
            createNodeSizeTable(model, feature_type)
            kernel = node_sort(model, kernel_budget, target_architecture)
        elif kernel_type == "path":
            createNodeSizeTable(model, feature_type)
            kernel = path_sort(model, kernel_budget, target_architecture)
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
    # generate code which looked the way I wanted it to look. Hence I sanatize the code
    # manually here. 
    # TODO Make this nicer and make sure that Labels: {} are correctly placed
    sanatized_impl = ""
    for line in implementation.split("\n"):
        if len(line) > 0 and not line.isspace():
            if line[0:4] == "    ":
                line = line[4:]
            sanatized_impl += line + "\n"

    with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
        out_file.write(sanatized_impl)

    with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
        out_file.write(header)

def getLeafTestCpp(node, feature_type):
        leafTest = """__attribute__((section("test_leaf_{nid}"))) void test{nid}({feature_type} const * const x, double * pred){\n"""\
        .replace("{nid}", str(node.id))\
        .replace("{feature_type}", feature_type)
        for i in range(len(node.prediction)):
            leafTest += "\tpred[{i}] += {prob};\n"\
            .replace("{i}", str(i))\
            .replace("{prob}", str(node.prediction[i]))
        leafTest += "\treturn;\n}\n"
        return leafTest
    
def getSplitTestCpp(node, feature_type):
    nidStr = ''
    compare = '0'
    if node.feature == 0:
        nidStr = '__'
        compare = '1'
    nidStr += str(node.id)
    splitTest = """__attribute__((section("test_split_{nid}"))) unsigned int test{nid}({feature_type} const * const x, double * pred){
        if( x[{cmp}] <= 20 ){
            if( x[{feature}] <= {value} ){
                return 10;
            }
            else { return 40; }
        }
        else{ return 30; }
    } \n"""\
    .replace("{nid}",nidStr)\
    .replace("{value}",str(node.split))\
    .replace("{feature}",str(node.feature))\
    .replace("{cmp}",compare)\
    .replace("{feature_type}", feature_type)
    return splitTest
    
def createNodeSizeTable(tree, feature_type):
    #initialize node size table
    tree.nodeSizeTable = []
    #empty section define
    cppStr = """__attribute__((section("leafEmpty"))) void emp({feature_type} const * const x, double * pred){} \n"""\
        .replace("{feature_type}", str(feature_type))
    cppStr += """__attribute__((section("splitReturnEmpty"))) unsigned int sp_return_emp({feature_type} const * const x, double * pred){
        return 40;
        } \n"""\
        .replace("{feature_type}", feature_type)
    cppStr += """__attribute__((section("splitEmpty"))) unsigned int sp_emp({feature_type} const * const x, double * pred){
        if( x[0] <= 20 ){
            return 10;
        }
        else{ return 30; }
    } \n"""\
    .replace("{feature_type}", feature_type)
    cppStr += """__attribute__((section("splitEmpty1"))) unsigned int sp_emp1({feature_type} const * const x, double * pred){
        if( x[1] <= 20 ){
            return 10;
        }
        else{ return 30; }
    } \n"""\
    .replace("{feature_type}", feature_type)
    #create section file
    for i in range(len(tree.nodes)):
        if tree.nodes[i].prediction is None:
            cppStr += getSplitTestCpp(tree.nodes[i], feature_type)
        else:
            cppStr += getLeafTestCpp(tree.nodes[i], feature_type)
    f = open("sizeOfNode.cpp", "w")
    f.write(cppStr)
    f.close()
    os.system("g++ sizeOfNode.cpp -c -std=c++11 -Wall -O3 -funroll-loops -ftree-vectorize")
    os.system("objdump -h sizeOfNode.o > sizeOfNode")
    #read sizeOfNode
    f = open("sizeOfNode", "r")
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
                    tree.nodeSizeTable.append([ int(x[1][10:]), int( x[2], 16) - leafE])
                elif x[1][5:12] == 'split__':
                    tree.nodeSizeTable.append([ int(x[1][13:]), int( x[2], 16) - splitE1])
                else:
                    tree.nodeSizeTable.append([ int(x[1][11:]), int( x[2], 16) - splitE])
            elif(x[1] == 'leafEmpty'):
                leafE = int( x[2], 16)
            elif(x[1] == 'splitEmpty'):
                splitE = int( x[2], 16) +spReturnE
            elif(x[1] == 'splitEmpty1'):
                splitE1 = int( x[2], 16) +spReturnE
            elif(x[1] == 'splitReturnEmpty'):
                spReturnE = int( x[2], 16) - leafE
    #os.system("rm sizeOfNode.cpp")
    os.system("rm sizeOfNode.o")
    os.system("rm sizeOfNode")

def searchNodeSizeTable(tree, id):
    return tree.nodeSizeTable[id][1]