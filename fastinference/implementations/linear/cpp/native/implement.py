import os
import numpy as np

from jinja2 import Environment, FileSystemLoader

def simplify_array(array):
    """Try to simplify the data type of an array

    Args:
        array: The array to simplify

    Returns:
        The simplified array
    """
    array = np.array(array)

    array_int = array.astype(int)
    if (array_int == array).all():
        minimum = np.min(array_int)
        maximum = np.max(array_int)
        if minimum >= -(2 ** 7) and maximum < 2 ** 7:
            array = array_int.astype(np.int8)
        elif minimum >= -(2 ** 15) and maximum < 2 ** 15:
            array = array_int.astype(np.int16)
        elif minimum >= -(2 ** 31) and maximum < 2 ** 31:
            array = array_int.astype(np.int32)
        else:
            array = array_int
    return array

def ctype(dtype):
    if dtype == "float32":
        return "float"
    elif dtype == "float64":
        return "double"
    elif dtype == "int8":
        return "signed char"
    elif dtype == "int16":
        return "signed short"
    elif dtype == "int32":
        return "signed int"
    elif dtype == "int64":
        return "signed long"
    else:
        # Only go for fixed-sizes data types as a last resort
        return str(dtype) + "_t"

def larger_datatype(dtype1, dtype2):
    types = [
        ["unsigned char", "uint8_t"],
        ["unsigned short", "uint16_t"],
        ["unsigned int", "uint32_t"],
        ["unsigned long", "uint64_t"],
        ["float"],
        ["double"]
    ]

    if dtype1 in types[0]:
        return dtype2
    
    if dtype1 in types[1] and dtype2 not in types[0]:
        return dtype2

    if dtype1 in types[2] and (dtype2 not in types[0] + types[1]):
        return dtype2

    if dtype1 in types[3] and (dtype2 not in types[0] + types[1] + types[2]):
        return dtype2

    if dtype1 in types[4] and (dtype2 not in types[0] + types[1] + types[2] + types[3]):
        return dtype2

    return dtype1

def to_implementation(model, out_path, out_name, weight = 1.0, namespace = "FAST_INFERENCE", feature_type = "double",label_type = "double", internal_type="double", infer_types = True, **kwargs ):
    """Generates a (native) C++ implementation of the given linear model. Native means that the coefficients of the linear model are stored in arrays which are then iterated in regular for-loops. You can use this implementation by simply passing :code:`"cpp.native"` to the implement, e.g.

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.json")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp.native")

    Args:
        model (Linear): The linear model to be implemented
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight.. Defaults to 1.0.
        namespace (str, optional): The namespace under which this model will be generated. Defaults to "FAST_INFERENCE".
        feature_type (str, optional): The data types of the input features. Defaults to "double".
        label_type (str, optional): The data types of the label. Defaults to "double".
    """

    if weight != 1.0:
        model.coef *= weight 
        model.intercept *= weight 

    if infer_types:
        model.coef = simplify_array(model.coef)
        model.intercept = simplify_array(model.intercept)

        internal_type = larger_datatype(ctype(model.coef.dtype), ctype(model.intercept.dtype))
        internal_type = larger_datatype(internal_type, feature_type)

    # simplify_array returns in np.arrays, but for the code generation we require lists
    model.coef = model.coef.T.tolist()
    model.intercept = model.intercept.tolist()
    
    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    implementation = env.get_template('base.j2').render(
        model = model,
        model_name = model.name,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        code_static = "",
        weight = weight,
        internal_type = internal_type
    )

    header = env.get_template('header.j2').render(
        model = model,
        model_name = model.name,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        model_weight = weight,
        internal_type = internal_type
    )

    with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
        out_file.write(implementation)

    with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
        out_file.write(header)