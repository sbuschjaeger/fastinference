from jinja2 import Environment, FileSystemLoader
import os
import pathlib
from json import JSONEncoder

import numpy as np

import logging
from itertools import chain

from onnx import numpy_helper

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def dynamic_import(class_path, attr_name):
    '''
    Dynamically import the class / attribute with the given name under the given class path
    '''
    import importlib 
    mod = importlib.import_module(class_path, package=__package__)
    clazz = getattr(mod, attr_name)
    return clazz

def get_tensor_shape(graph, name, verbose=False):
    """Find a tensor by name and returns its shape

    Args:
        graph: The ONNX graph
        name: The name of the tensor

    Returns:
        A list of ints describing the shape of the tensor
    """
    for tensor in chain(graph.input, graph.value_info):
        if verbose:
            print("CHECKING {} == {}".format(tensor.name, name))
        if tensor.name == name:
            if verbose:
                print("FOUND!")
                print(tensor)
                print(tensor.type.tensor_type)
            return [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]

def get_initializer(graph, name):
    """Find an initializer by name and returns it

    Args:
        graph: The ONNX graph
        name: The name of the initializer

    Returns:
        The initializer as numpy array
    """
    for initializer in graph.initializer:
        if initializer.name == name:
            return numpy_helper.to_array(initializer)

def get_attribute(node, name):
    """Find an attribute of a node by name and returns it

    The get the value from the resulting attribute one still has to access the matching property.
    A full list of properties can be found here: https://github.com/onnx/onnx/blob/master/docs/IR.md#attributes

    Args:
        node: The node of the ONNX graph
        name: The name of the attribute

    Returns:
        The attribute
    """
    for attribute in node.attribute:
        if attribute.name == name:
            return attribute

def get_constant(node):
    # TODO This is a weird call. Refactor this!
    return np.array(numpy_helper.to_array(get_attribute(node, "value").t).item())
