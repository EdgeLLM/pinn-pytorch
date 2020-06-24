from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import activations
import jax
# import flax
import jax.numpy as jnp

class FNN(flax.nn.Module):
    """Feed-forword neural network.
    """

    def __init__(
        self,
        layer_size,
        activation,
    ):
        super(FNN, self).__init__()
        self.layer_size = layer_size
        self.activation = activation

    def get_activation(self, inputs):
        return activations.get(inputs, self.activation) 

    def apply(self, x):
        for i in range(len(self.layer_size) - 2):
            x = flax.nn.Dense(x, features=layer_size[i+1])
            x = get_activation(x) 
        x = flax.nn.Dense(x, features=layer_size[-1])