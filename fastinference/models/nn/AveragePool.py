from .Layer import Layer
from fastinference.Util import get_attribute

class AvgPool2d(Layer):
    """Average Pooling

    Attributes:
        input_shape = [N, C, H, W]: The shape of the input tensor
        kernel_shape (2): The size of the kernel along each axis
        output_shape: The shape of the resulting output tensor, must match the result of the pooling
        strides (2): Stride along each spatial axis
        pads (4): Padding for the beginning and ending along each spatial axis
        count_include_pad (bool): Whether include pad pixels when calculating values for the edges
    """
    def __init__(self, graph, node, input_shape):

        kernel_shape = get_attribute(node, 'kernel_shape').ints
        strides = get_attribute(node, 'strides').ints
        pads = get_attribute(node, 'pads').ints
        count_include_pad = get_attribute(node, 'pads').i

        out_row = (input_shape[2] - kernel_shape[2] + pads[0] + pads[2]) // strides[0] + 1
        out_col = (input_shape[3] - kernel_shape[3] + pads[1] + pads[3]) // strides[1] + 1
        output_shape = (1, input_shape[1], int(out_row), int(out_col))
        # assert (input_shape[2] - kernel_shape[0] + pads[0] + pads[2]) // strides[0] + 1 == output_shape[2]
        # assert (input_shape[3] - kernel_shape[1] + pads[1] + pads[3]) // strides[1] + 1 == output_shape[3]

        self.kernel_shape = kernel_shape
        self.strides = strides
        self.pads = pads
        self.count_include_pad = count_include_pad
        super().__init__(input_shape, output_shape, "averagepool")
