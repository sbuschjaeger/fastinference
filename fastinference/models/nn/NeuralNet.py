import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
import onnxruntime

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
    """Constructs the appropriate layer from the given graph and node.

    Args:
        graph: The onnx graph.
        node: The current node.
        input_shape (tuple): The input shape of the current node

    Raises:
        NotImplementedError: Throws an error if there is no implementation for the current node available.

    Returns:
        Layer: The newly constructed layer. 
    """
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
    """A (simplified) neural network model. This class currently supports feed-forward multi-layer perceptrons as well as feed-forward convnets. In detail the following operations are supported
        
        - Linear Layer
        - Convolutional Layer
        - Sigmoid Activation
        - ReLU Activation
        - LeakyRelu Activation
        - MaxPool
        - AveragePool
        - LogSoftmax
        - LogSoftmax
        - Multiplication with a constant (Mul)
        - Reshape
        - BatchNormalization
    
    All layers are stored in :code:`self.layer` which is already order for execution. Additionally, the original onnx_model is stored in :code:`self.onnx_model`.

    This class loads ONNX files to build the internal computation graph. This can sometimes become a little tricky since the ONNX exporter work differently for each framework / version. In PyToch we usually use 

    .. code-block:: python

        dummy_x = torch.randn(1, x_train.shape[1], requires_grad=False)
        torch.onnx.export(model, dummy_x, os.path.join(out_path,name), training=torch.onnx.TrainingMode.PRESERVE, export_params=True,opset_version=11, do_constant_folding=True, input_names = ['input'],  output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

    **Important**: This class automatically merges "Constant -> Greater -> Constant -> Constant -> Where" operations into a single step layer. This is specifically designed to parse Binarized Neural Networks, but might be wrong for some types of networks. 
    """
    def __init__(self, path_to_onnx, accuracy = None, name = "model"):
        """Constructor of NeuralNet.

        Args:
            onnx_neural_net (str): Path to the onnx file.
            accuracy (float, optional): The accuracy of this tree on some test data. Can be used to verify the correctness of the implementation. Defaults to None.
            name (str, optional): The name of this model. Defaults to "Model".
        """
        self.layers = []

        self.onnx_model = load(path_to_onnx)
        checker.check_model(self.onnx_model)
        graph = shape_inference.infer_shapes(self.onnx_model).graph

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
                #fastinference.Util.get_tensor_shape(graph, graph.node[node_id + 1].input[0])
                layer = Step(input_shape, threshold, low, high)
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
        
        super().__init__(num_classes = n_classes, category = "neuralnet", name = name, accuracy = accuracy)

    def predict_proba(self,X):
        """Applies this NeuralNet to the given data and provides the predicted probabilities for each example in X. This function internally calls :code:`onnxruntime.InferenceSession` for inference.. 

        Args:
            X (numpy.array): A (N,d) matrix where N is the number of data points and d is the feature dimension. If X has only one dimension then a single example is assumed and X is reshaped via :code:`X = X.reshape(1,X.shape[0])`

        Returns:
            numpy.array: A (N, c) prediction matrix where N is the number of data points and c is the number of classes
        """
        if len(X.shape) == 1:
            X = X.reshape(1,X.shape[0])
        
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session = onnxruntime.InferenceSession(self.onnx_model, sess_options=opts)
        input_name = session.get_inputs()[0].name 
        return session.run([], {input_name: X})
    