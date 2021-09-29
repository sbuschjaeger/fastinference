from .Layer import Layer
from fastinference.Util import get_initializer

class Gemm(Layer):
    """General matrix multiplication

    For NHWC Layout the Fully Connected Layer is generally defined in ID Layout. The code generator
    uses DI regardless, because it should be faster in most cases (when I > D).

    The binary Gemm Layer only allows for weights and inputs -1 (False) and 1 (True). The weights and outputs
    are packed into ints of size binary_word_size. It works as follows:

        x = popcount(gemm_weight xnor previous_output) + gemm_bias
        x = 2 * x - binary_word_size

    The last step is necessary to revert the encoding of -1 as 0 (False).

    Attributes:
        input_shape = [N, I]: The dimension of the input tensor
        output_shape = [N, D]: The dimension of the resulting output tensor
        weight (D x I): The weights
        bias (D): The biases
    """
    def __init__(self, graph, node, input_shape):
        self.weight = get_initializer(graph, node.input[1])
        self.bias = get_initializer(graph, node.input[2])
        super().__init__(input_shape, (1, self.weight.shape[0]), "gemm")
