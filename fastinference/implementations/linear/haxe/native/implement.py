import os
import numpy as np

from jinja2 import Environment, FileSystemLoader
import subprocess

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

def sanatize_types(dtype):
    if dtype in ["float", "double", "float32", "float64", "Float"]:
        return "Float"
    else:
        return "Int"

def ctype(dtype):
    if dtype == "float32" or dtype == "float64":
        return "Float"
    else:
        return "Int"

def larger_datatype(dtype1, dtype2):
    if dtype1 == "Int" and dtype2 == "Int":
        return "Int"
    else:
        return "Float"

def to_implementation(model, out_path, out_name, weight = 1.0, package_name = "fastinference", feature_type = "Float", label_type = "Float", internal_type="Float", infer_types = True, language = "cpp", **kwargs ):
    """Generates a (native) C++ implementation of the given linear model. Native means that the coefficients of the linear model are stored in arrays which are then iterated in regular for-loops. You can use this implementation by simply passing :code:`"haxe.native"` to the implement, e.g.

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.json")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp.native")

    Args:
        model (Linear): The linear model to be implemented
        out_path (str): The folder in which the code will be stored.
        out_name (str): The filename under which the code will be stored.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight.. Defaults to 1.0.
        package_name (str, optional): The package_name under which this model will be generated. Defaults to "fastinference".
        feature_type (str, optional): The data types of the input features. Defaults to "Float".
        label_type (str, optional): The data types of the label. Defaults to "Float".
        infer_types (boolean, optional): If true, then data types are infereed based on the weights in the model. Defaults to True.
        language (str, optional): The language in which this model should be generated via haxe. Can be :code:`{"cpp", "js", "swf", "neko", "php", "cpp", "cs", "java", "jvm", "python", "lua", "hl", "cppia"}`. For more information see https://haxe.org/manual/compiler-usage.html. Defaults to "cpp".
    """
    assert language in ["cpp", "js", "swf", "neko", "php", "cpp", "cs", "java", "jvm", "python", "lua", "hl", "cppia"]

    feature_type = sanatize_types(feature_type)
    label_type = sanatize_types(label_type)
    internal_type = sanatize_types(internal_type)

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
        package_name = package_name.lower(),
        label_type = label_type,
        code_static = "",
        weight = weight,
        internal_type = internal_type
    )

    build = env.get_template('build.j2').render(
        model = model,
        model_name = model.name,
        language = language,
        package_name = package_name.lower(), 
    )

    haxe_tmp_path = os.path.join(out_path, package_name)
    if not os.path.exists(haxe_tmp_path):
        os.mkdir(haxe_tmp_path)

    with open(os.path.join(haxe_tmp_path, "{}.hx".format(model.name)), 'w') as out_file:
        out_file.write(implementation)

    with open(os.path.join(out_path, "build.hxml"), 'w') as out_file:
        out_file.write(build)

    p = subprocess.Popen(["haxe", "build.hxml"], cwd=out_path)
    p.wait()