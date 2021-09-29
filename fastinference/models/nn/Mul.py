from .Layer import Layer
from fastinference.Util import get_initializer

class Mul(Layer):
    def __init__(self, graph, node, input_shape):
        scale = get_initializer(graph, node.input[1])
        if scale.shape[0] == 1 or scale.shape == input_shape:
            self.scale = scale
            super().__init__(input_shape, input_shape, "mul")
        else:
            raise NotImplementedError("Scale layer can only handle 1 dimensional scaling parameters or matching input dimensions.")