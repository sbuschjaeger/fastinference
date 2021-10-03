
import os
import numpy as np
from jinja2 import Environment, FileSystemLoader

from fastinference.models.nn.Conv2D import Conv2D
from fastinference.models.nn.MaxPool import MaxPool2d
from fastinference.models.nn.BatchNorm import BatchNorm
from fastinference.models.nn.Activations import Step
from fastinference.models.nn.Reshape import Reshape
from fastinference.models.nn.Gemm import Gemm

def render(layer, **kwargs):
    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )
    
    code_alloc = ""
    if not isinstance(layer, Reshape):
        code_alloc = env.get_template('alloc.j2').render(output_shape = layer.output_shape, **kwargs)

    code_init = ""
    if isinstance(layer, (Conv2D, Gemm)):
        if isinstance(layer, Conv2D):
            weight = layer.weight.transpose((2, 3, 1, 0))
        else:
            weight = layer.weight

        code_init += env.get_template('init.j2').render(
            name='weight', 
            shape=weight.shape,
            value=weight.tolist(),
            output_shape = layer.output_shape,
            input_shape = layer.input_shape,
            **kwargs
        )

        code_init += env.get_template('init.j2').render(
            name='bias', 
            shape=layer.bias.shape, 
            value=layer.bias.tolist(),
            output_shape = layer.output_shape,
            input_shape = layer.input_shape,
            **kwargs
        )
    
    if isinstance(layer, BatchNorm):
        code_init += env.get_template('init.j2').render(
            name='bias', 
            shape=layer.bias.shape,
            value=layer.bias.tolist(),
            output_shape = layer.output_shape,
            input_shape = layer.input_shape,
            **kwargs
        )
        code_init += env.get_template('init.j2').render(
            name='scale', 
            shape=layer.scale.shape, 
            value=layer.scale.tolist(),
            output_shape = layer.output_shape,
            input_shape = layer.input_shape,
            **kwargs
        )

    if isinstance(layer, Step):
        code_predict = env.get_template(layer.name + '.j2').render(
            output_shape = layer.output_shape,
            input_shape = layer.input_shape,
            high = layer.high,
            low = layer.low,
            threshold = layer.threshold,
            **kwargs
        )
    elif isinstance(layer, (Conv2D, MaxPool2d)):
        code_predict = env.get_template(layer.name + '.j2').render(
            output_shape = layer.output_shape,
            input_shape = layer.input_shape,
            kernel_shape = layer.kernel_shape,
            strides = layer.strides,
            pads = layer.pads,
            **kwargs
        )
    else:
        code_predict = env.get_template(layer.name + '.j2').render(
            output_shape = layer.output_shape,
            input_shape = layer.input_shape,
            **kwargs
        )

    return code_alloc, code_init, code_predict

def to_implementation(model, out_path, out_name, weight = 1.0, align = 0, namespace = "FAST_INFERENCE", feature_type = "double", label_type = "double", internal_type = "double", **kwargs):
    """Generates a C++ implementation of the given NeuralNet model. This implementation provides a NHWC layout for the convolution layers basically resulting in a structure like this:

    .. code-block:: python

        for n = 1..N:
            for h = 1..H:
                for w = 1..W:
                    for c = 1..C:
                        //...

    You can use this implementation by simply passing :code:`"cpp.NHWC"` to implement:

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.onnx")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp.NHWC")

    Args:
        model (NeuralNet): The NeuralNet model to be implemented
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight. Defaults to 1.0.
        align (int, optional): If align > 0 then allocated memory will be aligned using __attribute__((aligned({{align}}))) where {{align}} is replaced by the given align value. If align = 0 then no memory alignment is performed. Defaults to 0.
        namespace (str, optional): The namespace under which this model will be generated. Defaults to "FAST_INFERENCE".
        feature_type (str, optional): The data types of the input features. Defaults to "double".
        label_type (str, optional): The data types of the label. Defaults to "double".
        internal_type (str, optional): The data type used for internal buffers and memory allocation. Defaults to "double".
    """

    input_type = feature_type

    # The code is generated in three parts and joined later to avoid multiple loops over the graph
    code_init = ''  # The C code containing initialized variables, i.e. weights and biases
    code_alloc = ''  # The C code allocating necessary variables, i.e. layer outputs
    code_predict = ''  # The C code containing the predict function

    flatten = None
    for layer_id, layer in enumerate(model.layers):
        # TODO We should not perform any changes on the layers directly to keep them separated
        if isinstance(layer, Reshape) and len(layer.output_shape) > 2:
            layer.output_shape = (layer.output_shape[0], *layer.output_shape[2:], layer.output_shape[1])

        if isinstance(layer, Reshape) and len(layer.input_shape) > len(layer.output_shape):
            flatten = layer

        output_type = input_type

        if flatten is not None:
            if isinstance(layer, BatchNorm):
                layer.scale = np.moveaxis(layer.scale.reshape(flatten.input_shape), -3, -1).reshape(flatten.output_shape)
                layer.bias = np.moveaxis(layer.bias.reshape(flatten.input_shape), -3, -1).reshape(flatten.output_shape)
            elif isinstance(layer, Step) and isinstance(layer.threshold, np.ndarray):
                layer.threshold = np.moveaxis(layer.threshold.reshape(flatten.input_shape), -3, -1).reshape(flatten.output_shape)
            elif isinstance(layer, Gemm):
                layer.weight = np.moveaxis(
                    layer.weight.reshape([layer.weight.shape[0]] + list(flatten.input_shape)),-3,-1
                ).reshape(layer.weight.shape)
                flatten = None

        a, i, p = render(
            layer,
            align = align, 
            feature_type = feature_type,
            namespace = namespace,
            label_type = label_type,
            internal_type = internal_type,
            layer_id = layer_id + 1, 
            input_type = input_type,
            output_type = output_type
        )
        code_init, code_alloc, code_predict = code_init + i, code_alloc + a, code_predict + p
        input_type = internal_type

    input_shape = model.layers[0].input_shape
    output_shape = model.layers[-1].output_shape[1:]

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    code_static = code_init + '\n' + code_alloc
    implementation = env.get_template('base.j2').render(
        model_name = model.name, 
        model_weight = weight, 
        in_layer_id = 0,
        out_layer_id = len(model.layers),
        input_shape = input_shape,
        output_shape = output_shape,
        code_predict = code_predict, 
        align = align, 
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        internal_type = internal_type,
        code_static = code_static
    )

    header = env.get_template("header.j2").render(
        model = model,
        align = align, 
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type,
        internal_type = internal_type,
        code_predict = code_predict, 
        code_static = code_static
    )

    with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
        out_file.write(implementation)

    with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
        out_file.write(header)    