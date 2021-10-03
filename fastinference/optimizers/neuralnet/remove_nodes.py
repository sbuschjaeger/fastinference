from fastinference.models.nn.Conv2D import Conv2D
from fastinference.models.nn.MaxPool import MaxPool2d
from fastinference.models.nn.BatchNorm import BatchNorm
from fastinference.models.nn.Activations import Sigmoid, Step
from fastinference.models.nn.Reshape import Reshape
from fastinference.models.nn.Gemm import Gemm
from fastinference.models.nn.AveragePool import AvgPool2d
from fastinference.models.nn.Activations import LogSoftmax, LeakyRelu, Relu, Sigmoid, Sign
from fastinference.models.nn.Mul import Mul

def optimize(model, **kwargs):
    """Removes LogSoftmax and positive scaling (Mul) layers from the network because they do not change the prediction.

    Args:
        model (NeuralNet): The NeuralNet model.

    Returns:
        NeuralNet: The NeuralNet model with removed layers.
    """
    while( (isinstance(model.layers[-1], Mul) and not isinstance(model.layers[-1].scale, tuple) and model.layers[-1].scale[0] > 0) or isinstance(model.layers[-1], LogSoftmax) ):
        model.layers.pop()
    
    return model