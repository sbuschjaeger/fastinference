import os
from sys import implementation
import numpy as np

from jinja2 import Environment, FileSystemLoader
from numpy.lib.arraysetops import isin

from iree import runtime as ireert
from iree.tf.support import module_utils
from iree.compiler import compile_str

from fastinference.models.nn.Activations import LeakyRelu
from fastinference.models.nn.BatchNorm import BatchNorm
from fastinference.models.nn.Conv2D import Conv2D
from fastinference.models.nn.Gemm import Gemm
from fastinference.models.nn.Reshape import Reshape

def to_implementation(model, out_path, out_name, weight = 1.0, package_name = "fastinference", feature_type = "f32",label_type = "f32", internal_type="f32", iree_backend = "iree_vmvx", **kwargs ):
    """Generates a TOSA implementation of the given neural net model and outputs a corresponding iree flatbuffer file for the specified backend. [TOSA](https://mlir.llvm.org/docs/Dialects/TOSA/) is an open source [MLIR](https://mlir.llvm.org/) dialect for tensor operations. This implementation transforms the given neural net to TOSA code which can then be compiled / executed the desired backend (e.g. CPU, GPU etc. ). Additionally, this implementation outputs a [iree](https://google.github.io/iree/) compatible flatbuffer file which can be executed with the iree runtime engine on the cpu (iree_vmvx, iree_llvmaot) or the gpu (iree_vulkan). There also seem to be support of tensorflow(lite) in iree, but this was not tested yet and throws an exception. This implementation uses a NCHW layout and does not perform any optimizations on its own.

    **Important**: Currently this implementation does not support quantization and thus setting feature_type / label_type / internal_type to anything unequal "f32" may lead to undesired behavior. Quantization support is planned.

    You can use this implementation by simply passing :code:`"iree"` to implement:

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.onnx")
        loaded_model.implement("/some/nice/place/", "mymodel", "iree")

    Args:
        model (NeuralNet): The NeuralNet model to be implemented
        out_path (str): The folder in which the :code:`*.mlir` and :code:`*.vm` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight. Defaults to 1.0.
        package_name (str, optional): The package_name under which this model will be generated. Defaults to "fastinference".
        feature_type (str, optional): The data types of the input features. Can be :code:`{"i8", "i16", "i32", "f32"}`. Defaults to "f32".
        label_type (str, optional): The data types of the label. Can be :code:`{"i8", "i16", "i32", "f32"}`. Defaults to "f32".
        internal_type (str, optional): The data type used for internal buffers and memory allocation. Can be :code:`{"i8", "i16", "i32", "f32"}`. Defaults to "f32".
        iree_backend (str, optional): The iree backend that is used for the compilation of the flatbuffer files. Can be :code:`{"iree_vmvx", "iree_llvmaot", "iree_vulkan"}`. Defaults to "iree_vmvx".

    Raises:
        ValueError: [description]
    """
    assert feature_type in ["i8", "i16", "i32", "f32"]
    assert label_type in ["i8", "i16", "i32", "f32"]
    assert internal_type in ["i8", "i16", "i32", "f32"]
    assert iree_backend in ["iree_vmvx", "iree_llvmaot", "iree_vulkan"]

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    for i in range(len(model.layers)):
        if isinstance(model.layers[i], LeakyRelu):
            raise ValueError("LeakyRelu is currently not supported by the `tosa` dialect (https://mlir.llvm.org/docs/Dialects/TOSA/) and hence not supported by iree. You can try a different backend, e.g. cpp.NHWC to compile this model.")

        # isinstance(model.layers[i], (Reshape, Conv2D, BatchNorm)) and 
        if len(model.layers[i].output_shape) > 2:
            # TODO THIS BREAKS FOR ResNet or any more complex models!
            model.layers[i].output_shape = (model.layers[i].output_shape[0], *model.layers[i].output_shape[2:], model.layers[i].output_shape[1])
            if i < len(model.layers) - 1:
                model.layers[i+1].input_shape = model.layers[i].output_shape

        if isinstance(model.layers[i], Gemm):
            model.layers[i].weight = model.layers[i].weight.tolist()
            model.layers[i].bias = model.layers[i].bias.tolist()
        
        if isinstance(model.layers[i], Conv2D):
            model.layers[i].weight = model.layers[i].weight.transpose((0, 2, 3, 1)).tolist()
            model.layers[i].bias = model.layers[i].bias.tolist()
            #model.layers[i].bias = model.layers[i].bias[None,None,:].tolist()
            model.layers[i].kernel_shape = list(model.layers[i].kernel_shape)

        if isinstance(model.layers[i], BatchNorm):
            model.layers[i].scale = model.layers[i].scale.tolist()
            model.layers[i].bias = model.layers[i].bias.tolist()

        model.layers[i].input_shape = list(model.layers[i].input_shape)
        model.layers[i].output_shape = list(model.layers[i].output_shape)


    implementation = env.get_template('base.j2').render(
        model = model,
        model_name = model.name,
        feature_type = feature_type,
        package_name = package_name.lower(),
        label_type = label_type,
        code_static = "",
        weight = weight,
        internal_type = internal_type, 
    )

    with open(os.path.join(out_path, "{}.{}".format(model.name,"mlir") ), 'w') as out_file:
        out_file.write(implementation)

    backend = module_utils.BackendInfo(iree_backend)
    compiled_flatbuffer = compile_str(implementation, input_type="tosa", target_backends=backend.compiler_targets)
    with open(os.path.join(out_path, "{}.{}".format(model.name,"vm") ), 'wb') as out_file:
        out_file.write(compiled_flatbuffer)

    # with open(os.path.join(out_path, "{}.{}".format(model.name,"mlir") ), 'r') as out_file:
        # implementation = out_file.read()

    # import sys
    # sys.exit(1)
    #x = np.random.rand(*model.layers[0].input_shape)    
    # x = np.float32(x)
    # print("x: ", x)
    # model_pred = np.array(model.predict_proba(x[0])[0])
    # print("MODEL PRED WAS: ", model_pred)
    # print("MODEL PRED WAS SHAPE: ", model_pred.shape)
    # vm_module = ireert.VmModule.from_flatbuffer(compiled_flatbuffer)

    # # Register the module with a runtime context.
    # config = ireert.Config(backend.driver)
    # ctx = ireert.SystemContext(config=config)
    # ctx.add_vm_module(vm_module)
    # predict = ctx.modules.module["{}_predict".format(model.name)]

    # #tmp = np.float32(x[None,None,:])
    # pred = predict(x)
    # print("IREE PRED WAS: ", pred)
    # print("IREE PRED WAS SHAPE: ", pred.shape)
    # DONE
    # with open(os.path.join(out_path, "{}.{}".format("build","hxml")), 'w') as out_file:
    #     out_file.write(build)