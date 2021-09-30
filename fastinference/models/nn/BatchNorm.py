import numpy as np
from .Layer import Layer
from fastinference.Util import get_initializer, get_attribute

class BatchNorm(Layer):
    """1D and 2D Batch normalization, depending on the given shape

    Scale and bias should already be transformed for inference according to https://arxiv.org/pdf/1502.03167.pdf

    Attributes:
        input_shape (tuple): The shape of the input tensor
        output_shape (tuple): The shape of the resulting output tensor, must match the input shape
        scale (float): The scale tensor
        bias (float): The bias tensor
    """
    def __init__(self, graph, node, input_shape):
        scale = get_initializer(graph, node.input[1])
        bias = get_initializer(graph, node.input[2])
        mean = get_initializer(graph, node.input[3])
        var = get_initializer(graph, node.input[4])
        epsilon = get_attribute(node, 'epsilon').f
        
        # Calculate scale and bias for inference according to the original paper
        # https://arxiv.org/abs/1502.03167
        self.bias = bias - scale * mean / np.sqrt(var + epsilon)
        self.scale = scale / np.sqrt(var + epsilon)
        super().__init__(input_shape, input_shape, "batchnorm")
