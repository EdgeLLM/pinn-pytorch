from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
# import flax 
import jax.numpy as jnp


def get(inputs, identifier):
    if isinstance(identifier, str):
        return {
            "elu": flax.nn.elu(inputs),
            "relu": flax.nn.relu(inputs), 
            "selu": flax.nn.selu(inputs),
            "sigmoid": flax.nn.sigmoid(inputs), 
            "sin": jnp.sin(inputs),
            "tanh": flax.nn.tanh(inputs),
        }[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret activation function identifier", identifier)