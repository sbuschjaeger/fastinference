import numpy as np

from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_shape, output_shape, name):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name