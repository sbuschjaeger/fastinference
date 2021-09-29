from itertools import islice
import logging, os
from math import ceil

import numpy as np

from onnx import helper, numpy_helper, utils, checker, load, shape_inference

from .Activations import Sigmoid, Relu, LeakyRelu, LogSoftmax, Step
from .Conv2D import Conv2D
from .Gemm import Gemm
from .MaxPool import MaxPool2d
from .AveragePool import AvgPool2d
from .Mul import Mul
from .BatchNorm import BatchNorm 
from .Reshape import Reshape
from ..Model import Model
import fastinference.Util #import simplify_array, to_c_data_type, get_tensor_shape, get_initializer, get_attribute, get_constant

def layer_from_node(graph, node, input_shape):
    if node.op_type == 'Conv':  # Convolution
        return Conv2D(graph, node, input_shape)
    elif node.op_type == 'BatchNormalization':
        return BatchNorm(graph, node, input_shape)
    elif node.op_type == 'Sigmoid':
        return Sigmoid(graph, node, input_shape)
    elif node.op_type == 'Relu':
        return Relu(graph, node, input_shape)
    elif node.op_type == 'LeakyRelu':
        return LeakyRelu(graph, node, input_shape)
    elif node.op_type == 'MaxPool':
        return MaxPool2d(graph, node, input_shape)
    elif node.op_type == 'AveragePool':
        return AvgPool2d(graph, node, input_shape)
    elif node.op_type == 'Gemm':  # Fully Connected
        return Gemm(graph, node, input_shape)
    elif node.op_type == "Mul":
        return Mul(graph, node, input_shape)
    elif node.op_type == 'LogSoftmax':
        return LogSoftmax(graph, node, input_shape)
    elif node.op_type == 'Reshape':
        return Reshape(graph, node, input_shape)
    else:
        raise NotImplementedError("Warning: Layer {} is currently not implemented".format(node.op_type))

class NeuralNet(Model):
    """A neural network model

    From the ONNX graph an internal model is constructed by iterating all layers.
    Greater/Less + Where is converted to a Step Layer. The Step Layer is merged with a subsequent
    BatchNormalization layer. Squeeze and Unsqueeze Layers are removed when the last dimension of
    the inbetween Layer has size 1 (as in PyTorch 2D BatchNormalization).

    Args:
        onnx_neural_net: The ONNX file name

    Attributes:
        name: The unique name of the graph for ensembles
        layers (list): The list of the layers assembling the network from input to output layer
    """
    def __init__(self, onnx_neural_net, model_accuracy = None, model_name = "model"):
        self.layers = []

        model = load(onnx_neural_net)
        checker.check_model(model)
        model = shape_inference.infer_shapes(model)
        graph = model.graph

        # Add implicit dimension for number of examples. We only consider single-example inference at the moment.
        input_shape = (1, graph.input[0].type.tensor_type.shape.dim[1].dim_value)
        n_classes = graph.output[0].type.tensor_type.shape.dim[1].dim_value

        # Iterate all layers in the ONNX graph and generate code depending on the layer type
        #node_iterator = iter(enumerate(graph.node))
        for node_id, node in enumerate(graph.node):
            print('Checking {}.'.format(node.op_type))

            if node.op_type == "Constant" and len(graph.node) > node_id + 5 and graph.node[node_id + 1].op_type == "Greater"\
                and graph.node[node_id + 2].op_type == "Constant" and graph.node[node_id + 3].op_type == "Constant" and graph.node[node_id + 4].op_type == "Where":
                print("Merging Constant {} -> Greater {} -> Constant {} -> Constant {} -> Where {} into Step layer".format(node_id, node_id + 1, node_id + 2, node_id + 3, node_id + 4))
                threshold = fastinference.Util.get_constant(node)
                high = fastinference.Util.get_constant(graph.node[node_id + 2])
                low = fastinference.Util.get_constant(graph.node[node_id + 3])

                # [dim.dim_value for dim in graph.node[node_id + 1].type.tensor_type.shape.dim]
                # graph.node[node_id + 1]
                layer = Step(fastinference.Util.get_tensor_shape(graph, graph.node[node_id + 1].input[0]), threshold, low, high)
            elif node.op_type in ['Constant', 'Shape', 'Slice', 'Concat', 'Gather', 'Greater', 'Where', 'Unsqueeze', 'Concat']:
                print('Layer {} skipped.'.format(node.op_type))
                continue
            else:
                try:
                    layer = layer_from_node(graph, node, input_shape)
                except Exception as e:
                    print(e)
                    print("Exception for node {} of type {} occured.".format(node_id, node.op_type))
            
            self.layers.append(layer)
            print("input shape is {}".format(input_shape))
            print("output shape is {}".format(layer.output_shape))
            print("")
            input_shape = layer.output_shape
        
        super().__init__(num_classes = n_classes, model_category = "neuralnet", model_name = model_name, model_accuracy = model_accuracy)

    def optimize(self):
        # If the last layer is a positive scale layer just remove it
        if isinstance(self.layers[-1], Mul) and not isinstance(self.layers[-1].scale, tuple) and self.layers[-1].scale[0] > 0:
            self.layers.pop()

        # Merge BN + Step layers for BNNs
        new_layers = []
        last_layer = None
        for layer_id, layer in enumerate(self.layers):
            if last_layer is not None and isinstance(last_layer, BatchNorm) and isinstance(layer, Step):
                layer.threshold = last_layer.scale * last_layer.mean / np.sqrt(last_layer.var + last_layer.epsilon)
            else:
                new_layers.append(last_layer)
            last_layer = layer

    # def to_implementation(self, out_path, out_name, backend, weight = 1.0):
    #     """Generate neural network code

    #     For each layer in the model, the render function is called to generate the C code.
    #     The code is generated in three parts (initialization, allocation, prediction).
    #     The code is stored to the attributes self.code_static (init + alloc), self.code_predict
    #     and self.includes. Before code generation, the neural network model is optimized.

    #     - The weights are quantized if backend['quantize'] is specified. This will
    #       invoke a clustering and assign each weight their cluster center.
    #     - The weights are simplified if backend['simplify_weights'] is True.
    #       They are converted to the lowest possible data type without losing precision.
    #     - Binary operations are used if backend['binary'] is specified. If it is True,
    #       the binary word size is inferred automatically from the model. Otherwise it
    #       is set to the specified value. Binary operations only allow for weights -1 and 1.
    #     - The memory layout might deviate form the standard specification to optimize
    #       inference time.

    #     Args:
    #         backend: The backend
    #         weight: The weight of the model in an ensemble
    #     """
    #     input_type = backend.feature_type

    #     # The code is generated in three parts and joined later to avoid multiple loops over the graph
    #     code_init = ''  # The C code containing initialized variables, i.e. weights and biases
    #     code_alloc = ''  # The C code allocating necessary variables, i.e. layer outputs
    #     code_predict = ''  # The C code containing the predict function

    #     #print("GENERATING CODE NOW!")
    #     last_flatten = None
    #     for layer_id, layer in enumerate(self.layers):
    #         # Flatten layer puts output in NHWC layout in wrong order for efficiency
    #         # Just put weights until next Gemm Layer in wrong order, too!
    #         if backend.implementation_type == 'NHWC' and isinstance(layer, Flatten):
    #             last_flatten = layer
    #         if last_flatten:
    #             if isinstance(layer, BatchNormalization):
    #                 layer.scale = np.moveaxis(layer.scale.reshape(last_flatten.input_shape), -3, -1).reshape(last_flatten.output_shape)
    #                 layer.bias = np.moveaxis(layer.bias.reshape(last_flatten.input_shape), -3, -1).reshape(last_flatten.output_shape)
    #             if isinstance(layer, Step) and isinstance(layer.threshold, np.ndarray):
    #                 layer.threshold = np.moveaxis(layer.threshold.reshape(last_flatten.input_shape), -3, -1).reshape(last_flatten.output_shape)
    #             if isinstance(layer, Gemm):
    #                 layer.weight = np.moveaxis(layer.weight.reshape([layer.weight.shape[0]] + list(last_flatten.input_shape)),-3,-1).reshape(layer.weight.shape)
    #                 last_flatten = None

    #         output_type = layer.output_type(input_type, backend)
    #         logging.info('Output type of layer {}: \t{}'.format(layer.__class__.__name__, output_type))

    #         i, a, p = layer.render(
    #             backend = backend, 
    #             layer_id=layer_id + 1, 
    #             input_type=input_type,
    #             output_type=to_c_data_type(output_type, backend)
    #         )
    #         code_init, code_alloc, code_predict = code_init + i, code_alloc + a, code_predict + p
    #         input_type = output_type

    #     input_shape = self.layers[0].input_shape
    #     output_shape = self.layers[-1].output_shape[1:]

    #     code_static = code_init + '\n' + code_alloc
    #     code_predict = Layer.render(
    #         'predict', 
    #         model_name = self.name, 
    #         model_weight = weight, 
    #         in_layer_id = 0,
    #         out_layer_id = len(self.layers),
    #         input_shape = input_shape,
    #         output_shape = output_shape,
    #         code_predict = code_predict, 
    #         backend = backend
    #     )
    #     includes = Layer.render('includes', backend = backend)
        
    #     implementation = env.get_template(os.path.join(backend.language ,"base.j2")).render(
    #         model = self,
    #         backend = backend, 
    #         includes = includes,
    #         code_predict = code_predict, 
    #         code_static = code_static
    #     )

    #     header = env.get_template(os.path.join(backend.language ,"header.j2")).render(
    #         model = self,
    #         backend = backend, 
    #         includes = includes,
    #         code_predict = code_predict, 
    #         code_static = code_static
    #     )

    #     if backend.language == "cpp":
    #         iext = "cpp"
    #         hext = "h"
    #     else:
    #         iext = ""
    #         hext = ""

    #     with open(os.path.join(out_path, "{}.{}".format(out_name,iext) ), 'w') as out_file:
    #         out_file.write(implementation)
    
    #     with open(os.path.join(out_path, "{}.{}".format(out_name,hext)), 'w') as out_file:
    #         out_file.write(header)
    
    def to_json_file(self):
        return self.path_name