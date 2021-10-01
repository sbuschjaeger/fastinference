from fastinference.models.nn.Conv2D import Conv2D
from fastinference.models.nn.MaxPool import MaxPool2d
from fastinference.models.nn.BatchNorm import BatchNorm
from fastinference.models.nn.Activations import Sigmoid, Step
from fastinference.models.nn.Reshape import Reshape
from fastinference.models.nn.Gemm import Gemm
from fastinference.models.nn.AveragePool import AvgPool2d
from fastinference.models.nn.Activations import LogSoftmax, LeakyRelu, Relu, Sigmoid, Sign

def optimize(model, **kwargs):
    # Merge BN + Step layers for BNNs
    new_layers = []
    last_layer = None
    for layer_id, layer in enumerate(model.layers):
        if last_layer is not None:
            if isinstance(last_layer, BatchNorm) and isinstance(layer, Step):
                layer.threshold = -last_layer.bias / last_layer.scale
            else:
                new_layers.append(last_layer)
        last_layer = layer
        
    new_layers.append(last_layer)
    model.layers = new_layers

    return model