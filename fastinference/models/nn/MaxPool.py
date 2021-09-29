from .Layer import Layer
from fastinference.Util import get_attribute

class MaxPool2d(Layer):
    """Maximum Pooling

    Binary MaxPool uses bitwise or to find the maximum bit value in each pooling window.

    Attributes:
        input_shape = [N, C, H, W]: The shape of the input tensor
        kernel_shape (2): The size of the kernel along each axis
        output_shape: The shape of the resulting output tensor, must match the result of the pooling
        strides (2): Stride along each spatial axis
        pads (4): Padding for the beginning and ending along each spatial axis
    """
    def __init__(self, graph, node, input_shape):
        assert len(input_shape) == 4
        kernel_shape = get_attribute(node, 'kernel_shape').ints
        strides = get_attribute(node, 'strides').ints
        pads = get_attribute(node, 'pads').ints
        out_row = (input_shape[2] - kernel_shape[0] + pads[0] + pads[2]) // strides[0] + 1
        out_col = (input_shape[3] - kernel_shape[1] + pads[1] + pads[3]) // strides[1] + 1
        output_shape = (1, input_shape[1], int(out_row), int(out_col))
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.pads =  pads
        super().__init__(input_shape, output_shape, "maxpool")