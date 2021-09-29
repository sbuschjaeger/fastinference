from .Layer import Layer
#from fastinference.Util import render, get_initializer, get_attribute
import fastinference.Util

class Conv2D(Layer):
    """Convolution

    The binary Convolution Layer uses HWMC instead of HWCM layout. The weights and outputs
    are packed into ints of size binary_word_size. The binary Convolution Layer only allows
    for weights and inputs -1 (False) and 1 (True). It works as follows:

        x = popcount(convolution_weight xnor previous_output) + convolution_bias
        x = 2 * x - binary_word_size

    The last step is necessary to revert the encoding of -1 as 0 (False).

    Attributes:
        input_shape = [N, C, H, W]: The shape of the input tensor
        kernel_shape = [M, C, kH, kW]: The shape of the convolution kernel
        output_shape: The shape of the resulting output tensor, must match the result of the convolution
        weight (M x C x kH x kW): The weight tensor that will be used in the convolutions
        bias (M): The bias to be added to the convolution
        strides (2): Stride along each spatial axis
        pads (4): Padding for the beginning and ending along each spatial axis
    """
    def __init__(self, graph, node, input_shape):
        assert len(input_shape) == 4
        weight = fastinference.Util.get_initializer(graph, node.input[1])
        bias = fastinference.Util.get_initializer(graph, node.input[2])
        kernel_shape = weight.shape
        strides = fastinference.Util.get_attribute(node, 'strides').ints
        pads = fastinference.Util.get_attribute(node, 'pads').ints

        out_row = (input_shape[2] - kernel_shape[2] + pads[0] + pads[2]) // strides[0] + 1
        out_col = (input_shape[3] - kernel_shape[3] + pads[1] + pads[3]) // strides[1] + 1
        output_shape = (1, kernel_shape[0], int(out_row), int(out_col))

        self.kernel_shape = kernel_shape
        self.weight, self.bias, self.strides, self.pads = weight, bias, strides, pads
        super().__init__(input_shape, output_shape, "conv2d")

    # def to_implementation(self, backend):
    #     code_global = ""

    #     code_init = render(
    #         'init', 
    #         name='weight', 
    #         shape=self.weight.shape,
    #         value=self.weight.tolist(),
    #         backend=backend,
    #     )
    #     code_init += render(
    #         'init', 
    #         name='bias', 
    #         shape=self.self.bias.shape, 
    #         value=self.self.bias.tolist(),
    #         backend=backend,
    #     )
    #     code_predict = render(
    #         'Conv2D',
    #         input_shape=self.input_shape, 
    #         output_shape=self.output_shape, 
    #         kernel_shape=self.kernel_shape,
    #         strides=self.strides, 
    #         pads=self.pads, backend=backend,
    #     )

    #     return code_global, code_init, code_predict