
import os
import numpy as np
from jinja2 import Environment, FileSystemLoader

import textwrap
from math import ceil

from numpy.lib.arraysetops import isin

from fastinference.models.nn.Conv2D import Conv2D
from fastinference.models.nn.MaxPool import MaxPool2d
from fastinference.models.nn.BatchNorm import BatchNorm
from fastinference.models.nn.Activations import Sigmoid, Step
from fastinference.models.nn.Reshape import Reshape
from fastinference.models.nn.Gemm import Gemm
from fastinference.models.nn.AveragePool import AvgPool2d
from fastinference.models.nn.Activations import LogSoftmax, LeakyRelu, Relu, Sigmoid, Sign

# TODO ADD unsigned data types
def simplify_array(array):
    """Try to simplify the data type of an array

    Args:
        array: The array to simplify

    Returns:
        The simplified array
    """
    if np.isscalar(array):
        return array
    else:
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

def infer_binary_wordsize(int_type):
    if int_type == "unsigned char" or int_type == "uint8_t":
        return 8
    elif int_type == "unsigned short" or int_type == "uint16_t":
        return 16
    elif int_type == "unsigned int" or int_type == "uint32_t":
        return 32
    elif int_type == "unsigned long" or int_type == "uint64_t":
        return 64
    else:
        raise ValueError("int_type must be one of {unsigned char, unsigned short, unsigned int, unsigned long, uint8_t, uint16_t, uint32_t, uint64_t} but supplied was {}.".format(int_type))

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

def render(layer, input_type, layer_id = 0, is_first = False, float_type = "double", int_type = "int", uint_type = "unsigned int", infer_types = False, align = 0,popcount = None):

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    if infer_types:
        if isinstance(layer, (Conv2D, Gemm)):
            layer.weight = simplify_array(layer.weight)
            layer.bias = simplify_array(layer.bias)

        if isinstance(layer, BatchNorm):
            layer.scale = simplify_array(layer.scale)
            layer.bias = simplify_array(layer.bias)
        
        if isinstance(layer, Step):
            if input_type not in ["float", "double"]:
                # All computations are integer and hence we do not need float values for the thresholds
                layer.threshold = np.ceil(layer.threshold) if layer.threshold_is_high else np.floor(layer.threshold)
            layer.threshold = simplify_array(layer.threshold)
            layer.low = simplify_array(layer.low)
            layer.high = simplify_array(layer.high)

    if is_first and isinstance(layer, (Gemm, Conv2D)):
        output_type = larger_datatype(input_type, ctype(layer.weight.dtype))
        output_type = larger_datatype(output_type, ctype(layer.bias.dtype))

        code_alloc = env.get_template('regular_alloc.j2').render(
            output_shape = layer.output_shape, 
            output_type = output_type,
            align = align,
            layer_id = layer_id
        )

        if isinstance(layer, Conv2D):
            weight = layer.weight.transpose((2, 3, 1, 0))
        else:
            weight = layer.weight

        code_init = ""
        code_init += env.get_template('regular_init.j2').render(
            name='weight', 
            shape=weight.shape,
            value=weight.astype(layer.weight.dtype).tolist(),
            data_type = ctype(layer.weight.dtype) if infer_types else float_type,
            layer_id = layer_id,
            align = align
        )

        code_init += env.get_template('regular_init.j2').render(
            name='bias', 
            shape=layer.bias.shape, 
            value=layer.bias.astype(layer.bias.dtype).tolist(),
            data_type = ctype(layer.bias.dtype) if infer_types else float_type,
            layer_id = layer_id,
            align = align
        )

        code_predict = env.get_template('regular_' + layer.name + '.j2').render(
            layer = layer,
            layer_id = layer_id
        )
    else:
        binary_word_size = infer_binary_wordsize(uint_type)
        if isinstance(layer, (LeakyRelu, Relu, Sigmoid, Sign, MaxPool2d, Reshape)):
            output_type = input_type
        elif isinstance(layer, (AvgPool2d, LogSoftmax)):
            output_type = float_type
        elif isinstance(layer, (Gemm, Conv2D)):
            output_type = larger_datatype(input_type, ctype(layer.weight.dtype))
            output_type = larger_datatype(output_type, ctype(layer.bias.dtype))
            output_type = int_type
        elif isinstance(layer, Step):
            output_type = uint_type

        code_alloc = ""

        if isinstance(layer, (MaxPool2d,Step)):
            code_alloc = env.get_template('alloc.j2').render(
                output_shape = layer.output_shape, 
                output_type = output_type,
                layer_id = layer_id,
                align = align,
                binary_word_size = binary_word_size
            )
        elif not isinstance(layer, Reshape):    
            code_alloc = env.get_template('regular_alloc.j2').render(
                output_shape = layer.output_shape, 
                output_type = output_type,
                layer_id = layer_id,
                align = align
            )

        code_init = ""
        if isinstance(layer, (Conv2D, Gemm)):
            if isinstance(layer, Conv2D):
                weight = layer.weight.transpose((2, 3, 0, 1))
            else:
                weight = layer.weight

            if not np.all( (weight == 1) | (weight == -1) ):
                print("Warning: Found values other than {-1,+1} in the weights. Is this planned?")
                weight[weight >= 0] = 1
                weight[weight < 0] = -1
                layer.weight = simplify_array(layer.weight)

            if not np.all( (layer.bias == 1) | (layer.bias == -1) ):
                print("Warning: Found values other than {-1,+1} in the bias. Is this planned?")
                layer.bias[layer.bias >= 0] = 1
                layer.bias[layer.bias < 0] = -1
                layer.bias = simplify_array(layer.bias)


            weight_binary = (weight + 1) // 2

            # Fill with zeros to make the array size divisible by binary_word_size. This will will push the remainder weights
            # to the most significant bits in the last packed int which matches the behaviour of the Step Layer
            next_higher_divisible = ceil(weight.shape[-1] / binary_word_size) * binary_word_size
            zeros = np.zeros(weight_binary.shape[:-1] + (next_higher_divisible - weight_binary.shape[-1],), dtype=weight_binary.dtype)
            weight_binary = np.append(weight_binary, zeros, axis=-1)
            weight_binary = weight_binary.astype(int)

            if isinstance(layer, Conv2D):
                # Pack a given number of weights (bits) into a single int
                weight_binary = [
                    [
                        [
                            [
                                hex(int(b, 2)) for b in textwrap.wrap(''.join([str(w) for w in weight_binary[i][j][k]]),binary_word_size)
                            ] for k in range(weight_binary.shape[2])
                        ] for j in range(weight_binary.shape[1])
                    ] for i in range(weight_binary.shape[0])
                ]

            else:
                #code_alloc = env.get_template('regular_alloc.j2').render(output_shape = layer.output_shape, binary_word_size = binary_word_size, **kwargs)

                # Pack a given number of weights (bits) into a single int
                weight_binary = [
                    [
                        hex(int(b, 2)) for b in textwrap.wrap(''.join([str(w) for w in weight_binary[i]]),binary_word_size)
                    ] for i in range(weight_binary.shape[0])
                ]

            code_init += env.get_template('init.j2').render(
                name='weight', 
                layer_id = layer_id,
                data_type = uint_type, 
                shape=weight.shape,
                value=weight_binary,
                binary_word_size = binary_word_size,
                align = align
            )

            code_init += env.get_template('regular_init.j2').render(
                name='bias', 
                shape=layer.bias.shape, 
                value=layer.bias.astype(int).tolist(),
                binary_word_size = binary_word_size,
                data_type = ctype(layer.bias.dtype) if infer_types else int_type,
                layer_id = layer_id,
                align = align
            )
        
        if isinstance(layer, Step):
            code_init += env.get_template('regular_init.j2').render(
                name='threshold', 
                shape=layer.threshold.shape,
                value=layer.threshold.tolist(),
                binary_word_size = binary_word_size,
                data_type = ctype(layer.threshold.dtype) if infer_types else float_type,
                layer_id = layer_id,
                align = align
            )
        
        if popcount is None:
            if binary_word_size <= 32:
                popcount = "__builtin_popcount"
            else:
                popcount = "__builtin_popcountll"

        code_predict = env.get_template(layer.name + '.j2').render(
            layer = layer,
            binary_word_size = binary_word_size,
            layer_id = layer_id,
            align = align,
            int_type = int_type,
            uint_type = uint_type,
            float_type = float_type,
            popcount = popcount,
            output_type = output_type,
            input_type = input_type
        )

    return code_alloc, code_init, code_predict, output_type

def to_implementation(model, out_path, out_name, weight = 1.0, namespace = "FAST_INFERENCE", align = 0, feature_type = "double", label_type = "double", float_type = "double", int_type = "signed int", uint_type = "unsigned int", infer_types = True, popcount = None, **kwargs):
    """Generates a C++ implementation of the given binarized NeuralNet model by using the XNOR and popcount operations whenever possible. When infer_types is true, this implementation tries to infer the smallest possible data type which still guarantees a correct execution. Otherwise the supplied data types are used.  Note that the first layer is never binarized because we do not assume the input data to be in a packed integer format, but to be a regular array. 
    
    **Important**: This implementation performs basic optimization on the model. It 

        - removes unnecessary layers (see :meth:`fastinference.optimizers.neuralnet.remove_nodes.optimize`), 
        - merges BatchNorm + Step layers (see :meth:`fastinference.optimizers.neuralnet.merge_nodes.optimize`)
        - it maps all Conv and linear weights / biases to {-1, +1} if not done already
    
    **Important**: This implementation generates gcc compliant code by using the :code:`__builtin_popcount` or :code:`__builtin_popcountll` (depending on the uint_type) intrinsic. This intrinsic can vary from compiler to compiler. If you want to compile with another compiler (e.g. MSVC or clang) then you can supply the corresponding popcount operation via the popcount argument. However, please keep in mind that you might have to adapt the included headers manually after the code generation and that the popcount operation should fit the corresponding uint_type.

    You can use this implementation by simply passing :code:`"cpp.binary"` to implement:

    .. code-block:: python

        loaded_model = fastinference.Loader.model_from_file("/my/nice/model.onnx")
        loaded_model.implement("/some/nice/place/", "mymodel", "cpp.binary")

    References:

    Args:
        model (NeuralNet): The NeuralNet model to be implemented
        out_path (str): The folder in which the :code:`*.cpp` and :code:`*.h` files are stored.
        out_name (str): The filenames.
        weight (float, optional): The weight of this model inside an ensemble. The weight is ignored if it is 1.0, otherwise the prediction is scaled by the respective weight. Defaults to 1.0.
        align (int, optional): If align > 0 then allocated memory will be aligned using __attribute__((aligned({{align}}))) where {{align}} is replaced by the given align value. If align = 0 then no memory alignment is performed. Defaults to 0.
        namespace (str, optional): The namespace under which this model will be generated. Defaults to "FAST_INFERENCE".
        feature_type (str, optional): The data types of the input features. Defaults to "double".
        label_type (str, optional): The data types of the label. Defaults to "double".
        float_type (str, optional): The floating point type used when required. Defaults to "double".
        int_type (str, optional): The signed integer type used when required. Defaults to "signed int".
        uint_type (str, optional): The unsigned integer type used when required. Defaults to "unsigned int".
        infer_types (bool, optional): If True tries to infer the smallest possible data type which still guarantees a correct implementation. Defaults to True.
        popcount (str, optional): The popcount operation which should be used to compute popcount. If this is None, then :code:`__builtin_popcount` or :code:`__builtin_popcountll` is used depending on the binary_word_size required by the uint_type. Defaults to None.
    """

    model.optimize(["remove_nodes", "merge_nodes"], None)

    code_init = ""
    code_alloc = ""
    code_predict = ""

    flatten = None
    input_type = feature_type
    is_first = True
    for layer_id, layer in enumerate(model.layers):
        print("IMPLEMENTING {}".format(layer.name))

        if isinstance(layer, Reshape) and len(layer.output_shape) > 2:
            layer.output_shape = (layer.output_shape[0], *layer.output_shape[2:], layer.output_shape[1])

        if isinstance(layer, Reshape) and len(layer.input_shape) > len(layer.output_shape):
            flatten = layer

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

        a, i, p, input_type = render(
            layer,
            is_first = is_first,
            layer_id = layer_id + 1, 
            align = align,
            float_type = float_type,
            int_type = int_type,
            uint_type = uint_type,
            infer_types = infer_types,
            input_type = input_type,
            popcount=popcount
        )

        if isinstance(layer, (Gemm, Conv2D)):
            is_first = False

        code_init, code_alloc, code_predict = code_init + i, code_alloc + a, code_predict + p

    env = Environment(
        loader=FileSystemLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)))),
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    code_static = code_init + '\n' + code_alloc
    implementation = env.get_template('base.j2').render(
        name = model.name, 
        weight = weight, 
        in_layer_id = 0,
        out_layer_id = len(model.layers),
        feature_type = feature_type,
        code_predict = code_predict, 
        namespace = namespace,
        code_static = code_static,
        label_type = label_type,
        n_classes = model.n_classes
    )

    header = env.get_template("header.j2").render(
        model = model,
        accuracy = model.accuracy,
        feature_type = feature_type,
        namespace = namespace,
        label_type = label_type
    )

    with open(os.path.join(out_path, "{}.{}".format(out_name,"cpp") ), 'w') as out_file:
        out_file.write(implementation)

    with open(os.path.join(out_path, "{}.{}".format(out_name,"h")), 'w') as out_file:
        out_file.write(header)    