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

def sanatize_types(dtype):
    if dtype in ["float", "double", "float32", "float64"]:
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

def to_implementation(model, out_path, out_name, weight = 1.0, package_name = "fastinference", feature_type = "Float",label_type = "Float", internal_type="Float", infer_types = True, **kwargs ):
    """Generates a (native) C++ implementation of the given linear model. Native means that the coefficients of the linear model are stored in arrays which are then iterated in regular for-loops. You can use this implementation by simply passing :code:`"cpp.native"` to the implement, e.g.

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.json")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp.native")

    Args:
        model (Linear): The linear model to be implemented
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight.. Defaults to 1.0.
        package_name (str, optional): The package_name under which this model will be generated. Defaults to "fastinference".
        feature_type (str, optional): The data types of the input features. Defaults to "Float".
        label_type (str, optional): The data types of the label. Defaults to "Float".
    """

    feature_type = sanatize_types(feature_type)
    label_type = sanatize_types(label_type)
    internal_type = sanatize_types(internal_type)

    if infer_types:
        model.coef = simplify_array(model.coef)
        model.intercept = simplify_array(model.intercept)

        internal_type = larger_datatype(ctype(model.coef.dtype), ctype(model.intercept.dtype))
        internal_type = larger_datatype(internal_type, feature_type)

        # simplify_array returns in np.arrays, but for the code generation we require lists
        #model.coef = model.coef.tolist()
        #model.intercept = model.intercept.tolist()

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    #print(model.coef)
    # import copy
    # weights = np.array(copy.copy(model.coef))#[:,:,None]
    # print(weights.shape)
    # weights = np.swapaxes(weights, 0, 1)
    # print(weights.shape)

    model.coef = np.array([[1.0,0.0], [0.0,1.0]])
    model.intercept = np.array([0.0,0.0])
    model.num_classes = 2

    # TODO handle data types correctly
    dim = model.coef.shape[0]
    weights = model.coef

    implementation = env.get_template('base.j2').render(
        model = model,
        model_name = model.name,
        feature_type = feature_type,
        package_name = package_name.lower(),
        label_type = label_type,
        code_static = "",
        weight = weight,
        internal_type = internal_type, 
        weights = weights.T.tolist(), 
        bias = model.intercept.tolist(),
        dim = dim
    )

    # build = env.get_template('build.j2').render(
    #     model = model,
    #     model_name = model.name,
    #     language = "cpp",
    #     package_name = package_name.lower()
    # )

    with open(os.path.join(out_path, "{}.{}".format(model.name,"mlir") ), 'w') as out_file:
        out_file.write(implementation)


    x = np.random.rand(dim)    
    print("x: ", x)
    model_pred = model.predict_proba(x)
    print("MODEL PRED WAS: ", model_pred)

    from iree import runtime as ireert
    from iree.tf.support import module_utils
    from iree.compiler import compile_str
    from iree.compiler import tf as tfc

    backend_choice = "iree_vmvx (CPU)" #@param [ "iree_vmvx (CPU)", "iree_llvmaot (CPU)", "iree_vulkan (GPU/SwiftShader)" ]
    backend_choice = backend_choice.split(" ")[0]
    backend = module_utils.BackendInfo(backend_choice)

    #tosa_mlir = open("RidgeClassifier.mlir").read()
    compiled_flatbuffer = compile_str(implementation, input_type="tosa", target_backends=["vmvx"])

    vm_module = ireert.VmModule.from_flatbuffer(compiled_flatbuffer)

    # Register the module with a runtime context.
    config = ireert.Config(backend.driver)
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(vm_module)
    predict = ctx.modules.module["{}_predict".format(model.name)]

    #tmp = np.float32(x[None,None,:])
    tmp = np.float32(x)
    print(tmp.shape)
    print(tmp)
    print(tmp.dtype)
    pred = predict(tmp)
    print("IREE PRED WAS: ", pred)
    # DONE
    # with open(os.path.join(out_path, "{}.{}".format("build","hxml")), 'w') as out_file:
    #     out_file.write(build)