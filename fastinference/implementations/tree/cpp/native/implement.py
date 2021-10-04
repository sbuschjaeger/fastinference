import os
import numpy as np

from jinja2 import Environment, FileSystemLoader

def to_implementation(model, out_path, out_name, weight = 1.0, namespace = "FAST_INFERENCE", feature_type = "double", label_type = "double", int_type = "unsigned int", round_splits = False, infer_types = False, **kwargs):
    """Generates a native C++ implementation of the given Tree model. Native means that the tree is represented in an array structure which is iterated via a while-loop. You can use this implementation by simply passing :code:`"cpp.native"` to the implement, e.g.

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.json")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp.native")

    Args:
        model (Tree): The Tree model to be implemented
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight.. Defaults to 1.0.
        namespace (str, optional): The namespace under which this model will be generated. Defaults to "FAST_INFERENCE".
        feature_type (str, optional): The data types of the input features. Defaults to "double".
        label_type (str, optional): The data types of the label. Defaults to "double".
        round_splits (bool, optional): If True then all splits are rounded towards the next integer. Defaults to False,
        infer_types (bool, optional): If True then the smallest data type for index variables is inferred by the overall tree size. Otherwise "unsigned int" is used. Defaults to False,
    """
    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    if round_splits:
        for n in model.nodes:
            if n.prediction is None:
                n.split = np.ceil(n.split).astype(int)
    
    leaf_nodes = []
    inner_nodes = []
    to_expand = [model.nodes[0]]

    # Make sure that the nodes are correctly numbered given their current order
    while( len(to_expand) > 0 ):
        n = to_expand.pop(0)

        if n.prediction is not None:
            leaf_nodes.append(n)
        else:
            inner_nodes.append(n)
            n.leftChild.id = 2*len(inner_nodes) - 1
            n.rightChild.id = 2*len(inner_nodes)
            to_expand.append(n.leftChild)
            to_expand.append(n.rightChild)
    
    if infer_types:
        if len(inner_nodes) < 2**8:
            int_type = "unsigned char"
        elif len(inner_nodes) < 2**16:
            int_type = "unsigned short"
        elif len(inner_nodes) < 2**32:
            int_type = "unsigned int"
        else:
            int_type = "unsigned long"

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
        inner_nodes = inner_nodes
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
        inner_nodes = inner_nodes
    )

    with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
        out_file.write(implementation)

    with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
        out_file.write(header)