import numpy as np
from .Layer import Layer
from fastinference.Util import get_initializer

class Reshape(Layer):
    def __init__(self, graph, node, input_shape):
        for n in graph.node:
            if n.output[0] == node.input[1]:
                concat_node = n
                break

        output_shape = [input_shape[0]]
        for i in range(len(concat_node.input) - 1):
            output_shape.append( get_initializer(graph, concat_node.input[i+1])[0] )

        output_shape = tuple(output_shape)
        
        if len(output_shape) == 2 and output_shape[1] == -1:
            flatten_dim = np.prod( (input_shape) )
            output_shape = (1, flatten_dim)

        super().__init__(input_shape, output_shape, "reshape")