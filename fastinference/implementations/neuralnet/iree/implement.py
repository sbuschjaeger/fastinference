import os
from sys import implementation
import numpy as np

from jinja2 import Environment, FileSystemLoader
from numpy.lib.arraysetops import isin

from fastinference.models.nn.Activations import LeakyRelu
from fastinference.models.nn.BatchNorm import BatchNorm
from fastinference.models.nn.Conv2D import Conv2D
from fastinference.models.nn.Gemm import Gemm
from fastinference.models.nn.Reshape import Reshape

# def render(layer, **kwargs):
#     env = Environment(
#         loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
#         trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
#     )
    
#     code_alloc = ""
#     if not isinstance(layer, Reshape):
#         code_alloc = env.get_template('alloc.j2').render(output_shape = layer.output_shape, **kwargs)

#     code_init = ""
#     if isinstance(layer, (Conv2D, Gemm)):
#         if isinstance(layer, Conv2D):
#             weight = layer.weight.transpose((2, 3, 1, 0))
#         else:
#             weight = layer.weight

#         code_init += env.get_template('init.j2').render(
#             name='weight', 
#             shape=weight.shape,
#             value=weight.tolist(),
#             output_shape = layer.output_shape,
#             input_shape = layer.input_shape,
#             **kwargs
#         )

#         code_init += env.get_template('init.j2').render(
#             name='bias', 
#             shape=layer.bias.shape, 
#             value=layer.bias.tolist(),
#             output_shape = layer.output_shape,
#             input_shape = layer.input_shape,
#             **kwargs
#         )
    
#     if isinstance(layer, BatchNorm):
#         code_init += env.get_template('init.j2').render(
#             name='bias', 
#             shape=layer.bias.shape,
#             value=layer.bias.tolist(),
#             output_shape = layer.output_shape,
#             input_shape = layer.input_shape,
#             **kwargs
#         )
#         code_init += env.get_template('init.j2').render(
#             name='scale', 
#             shape=layer.scale.shape, 
#             value=layer.scale.tolist(),
#             output_shape = layer.output_shape,
#             input_shape = layer.input_shape,
#             **kwargs
#         )

#     if isinstance(layer, Step):
#         code_predict = env.get_template(layer.name + '.j2').render(
#             output_shape = layer.output_shape,
#             input_shape = layer.input_shape,
#             high = layer.high,
#             low = layer.low,
#             threshold = layer.threshold,
#             **kwargs
#         )
#     elif isinstance(layer, (Conv2D, MaxPool2d)):
#         code_predict = env.get_template(layer.name + '.j2').render(
#             output_shape = layer.output_shape,
#             input_shape = layer.input_shape,
#             kernel_shape = layer.kernel_shape,
#             strides = layer.strides,
#             pads = layer.pads,
#             **kwargs
#         )
#     else:
#         code_predict = env.get_template(layer.name + '.j2').render(
#             output_shape = layer.output_shape,
#             input_shape = layer.input_shape,
#             **kwargs
#         )

#     return code_alloc, code_init, code_predict

# def to_implementation(model, out_path, out_name, weight = 1.0, align = 0, namespace = "FAST_INFERENCE", feature_type = "double", label_type = "double", internal_type = "double", **kwargs):
#     """Generates a C++ implementation of the given NeuralNet model. This implementation provides a NHWC layout for the convolution layers basically resulting in a structure like this:

#     .. code-block:: python

#         for n = 1..N:
#             for h = 1..H:
#                 for w = 1..W:
#                     for c = 1..C:
#                         //...

#     You can use this implementation by simply passing :code:`"cpp.NHWC"` to implement:

#     .. code-block:: python

#         loaded_model = fastinference.Loader.model_from_file("/my/nice/model.onnx")
#         loaded_model.implement("/some/nice/place/", "mymodel", "cpp.NHWC")

#     Args:
#         model (NeuralNet): The NeuralNet model to be implemented
#         out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
#         out_name (str): The filenames.
#         weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight. Defaults to 1.0.
#         align (int, optional): If align > 0 then allocated memory will be aligned using __attribute__((aligned({{align}}))) where {{align}} is replaced by the given align value. If align = 0 then no memory alignment is performed. Defaults to 0.
#         namespace (str, optional): The namespace under which this model will be generated. Defaults to "FAST_INFERENCE".
#         feature_type (str, optional): The data types of the input features. Defaults to "double".
#         label_type (str, optional): The data types of the label. Defaults to "double".
#         internal_type (str, optional): The data type used for internal buffers and memory allocation. Defaults to "double".
#     """

#     input_type = feature_type

#     # The code is generated in three parts and joined later to avoid multiple loops over the graph
#     code_init = ''  # The C code containing initialized variables, i.e. weights and biases
#     code_alloc = ''  # The C code allocating necessary variables, i.e. layer outputs
#     code_predict = ''  # The C code containing the predict function

#     flatten = None
#     for layer_id, layer in enumerate(model.layers):
#         # TODO We should not perform any changes on the layers directly to keep them separated
#         if isinstance(layer, Reshape) and len(layer.output_shape) > 2:
#             layer.output_shape = (layer.output_shape[0], *layer.output_shape[2:], layer.output_shape[1])

#         if isinstance(layer, Reshape) and len(layer.input_shape) > len(layer.output_shape):
#             flatten = layer

#         output_type = input_type

#         if flatten is not None:
#             if isinstance(layer, BatchNorm):
#                 layer.scale = np.moveaxis(layer.scale.reshape(flatten.input_shape), -3, -1).reshape(flatten.output_shape)
#                 layer.bias = np.moveaxis(layer.bias.reshape(flatten.input_shape), -3, -1).reshape(flatten.output_shape)
#             elif isinstance(layer, Step) and isinstance(layer.threshold, np.ndarray):
#                 layer.threshold = np.moveaxis(layer.threshold.reshape(flatten.input_shape), -3, -1).reshape(flatten.output_shape)
#             elif isinstance(layer, Gemm):
#                 layer.weight = np.moveaxis(
#                     layer.weight.reshape([layer.weight.shape[0]] + list(flatten.input_shape)),-3,-1
#                 ).reshape(layer.weight.shape)
#                 flatten = None

#         a, i, p = render(
#             layer,
#             align = align, 
#             feature_type = feature_type,
#             namespace = namespace,
#             label_type = label_type,
#             internal_type = internal_type,
#             layer_id = layer_id + 1, 
#             input_type = input_type,
#             output_type = output_type
#         )
#         code_init, code_alloc, code_predict = code_init + i, code_alloc + a, code_predict + p
#         input_type = internal_type

#     input_shape = model.layers[0].input_shape
#     output_shape = model.layers[-1].output_shape[1:]

#     env = Environment(
#         loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
#         trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
#     )

#     code_static = code_init + '\n' + code_alloc
#     implementation = env.get_template('base.j2').render(
#         model_name = model.name, 
#         model_weight = weight, 
#         in_layer_id = 0,
#         out_layer_id = len(model.layers),
#         input_shape = input_shape,
#         output_shape = output_shape,
#         code_predict = code_predict, 
#         align = align, 
#         feature_type = feature_type,
#         namespace = namespace,
#         label_type = label_type,
#         internal_type = internal_type,
#         code_static = code_static
#     )

#     header = env.get_template("header.j2").render(
#         model = model,
#         align = align, 
#         feature_type = feature_type,
#         namespace = namespace,
#         label_type = label_type,
#         internal_type = internal_type,
#         code_predict = code_predict, 
#         code_static = code_static
#     )

#     with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
#         out_file.write(implementation)

#     with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
#         out_file.write(header)    



def to_implementation(model, out_path, out_name, weight = 1.0, package_name = "fastinference", feature_type = "Float",label_type = "Float", internal_type="Float", infer_types = True, **kwargs ):

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    # TODO handle data types correctly
    # dim = model.coef.shape[0]
    # weights = model.coef

    for i in range(len(model.layers)):
        if isinstance(model.layers[i], LeakyRelu):
            raise ValueError("LeakyRelu is currently not supported by the `tosa` dialect (https://mlir.llvm.org/docs/Dialects/TOSA/) and hence not supported by iree. You can try a different backend, e.g. cpp.NHWC to compile this model.")

        # if isinstance(layer, Reshape) and len(layer.input_shape) > len(layer.output_shape):
        #     flatten = layer

        # output_type = input_type

        # if flatten is not None:
        #     if isinstance(layer, BatchNorm):
        #         layer.scale = np.moveaxis(layer.scale.reshape(flatten.input_shape), -3, -1).reshape(flatten.output_shape)
        #         layer.bias = np.moveaxis(layer.bias.reshape(flatten.input_shape), -3, -1).reshape(flatten.output_shape)
        #     elif isinstance(layer, Step) and isinstance(layer.threshold, np.ndarray):
        #         layer.threshold = np.moveaxis(layer.threshold.reshape(flatten.input_shape), -3, -1).reshape(flatten.output_shape)
        #     elif isinstance(layer, Gemm):
        #         layer.weight = np.moveaxis(
        #             layer.weight.reshape([layer.weight.shape[0]] + list(flatten.input_shape)),-3,-1
        #         ).reshape(layer.weight.shape)
        #         flatten = None

        # isinstance(model.layers[i], (Reshape, Conv2D, BatchNorm)) and 
        if len(model.layers[i].output_shape) > 2:
            # TODO THIS BREAKS FOR REsNEt or any more complex models!
            model.layers[i].output_shape = (model.layers[i].output_shape[0], *model.layers[i].output_shape[2:], model.layers[i].output_shape[1])
            if i < len(model.layers) - 1:
                model.layers[i+1].input_shape = model.layers[i].output_shape

        if isinstance(model.layers[i], Gemm):
            # DEBUG
            prev = model.layers[-1]
            # print(model.layers[i].weight.shape)
            # model.layers[i].weight = np.moveaxis(
            #     #model.layers[i].weight.reshape([model.layers[i].weight.shape[0]] + list(prev.input_shape)),-3,-1
            #     model.layers[i].weight, 0,1
            # ).reshape(model.layers[i].weight.shape)
            # END DEBUG

            model.layers[i].weight = model.layers[i].weight.tolist()
            model.layers[i].bias = model.layers[i].bias.tolist()

            # model.layers[i].weight = model.layers[i].weight[None,:,:].swapaxes(1,2).tolist()
            # model.layers[i].bias = model.layers[i].bias[None,None,:].tolist()
        
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

        # if isinstance(model.layers[i], BatchNorm):
        #     if len(model.layers[i].scale.shape) == 1:
        #         model.layers[i].scale = model.layers[i].scale[None,None,:].tolist()
        #         model.layers[i].bias = model.layers[i].bias[None,None,:].tolist()
        #     elif len(model.layers[i].scale.shape) == 2:
        #         model.layers[i].scale = model.layers[i].scale[None,:,:].tolist()
        #         model.layers[i].bias = model.layers[i].bias[None,:,:].tolist()
        #     else:
        #         model.layers[i].scale = model.layers[i].scale.tolist()
        #         model.layers[i].bias = model.layers[i].bias.tolist()


        # if hasattr(model.layers[i], "weight"):
        #     model.layers[i].weight = model.layers[i].weight.tolist()
        
        # if hasattr(model.layers[i], "bias"):
        #     model.layers[i].bias = model.layers[i].bias.tolist()

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

    # with open(os.path.join(out_path, "{}.{}".format(model.name,"mlir") ), 'r') as out_file:
        # implementation = out_file.read()

    # import sys
    # sys.exit(1)
    x = np.random.rand(*model.layers[0].input_shape)    
    #x = np.array( [[[0.0624079,0.4068808,0.52959085,0.90529114,0.512078,0.03346087,0.16243644,0.1209345,0.04925144,0.1133609]]] )
    x = np.float32(x)
    print("x: ", x)
    model_pred = np.array(model.predict_proba(x[0])[0])
    print("MODEL PRED WAS: ", model_pred)
    print("MODEL PRED WAS SHAPE: ", model_pred.shape)

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
    pred = predict(x)
    print("IREE PRED WAS: ", pred)
    print("IREE PRED WAS SHAPE: ", pred.shape)
    # DONE
    # with open(os.path.join(out_path, "{}.{}".format("build","hxml")), 'w') as out_file:
    #     out_file.write(build)