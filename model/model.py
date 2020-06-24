from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data import Data
import jax
from jax import device_put
import flax
from flax import nn
import jax.numpy as jnp

class FNN(nn.Module):
    """Feed-forword neural network.
    """

    def apply(self, x):
        x = nn.Dense(x, features=50)
        x = nn.tanh(x)
        x = nn.Dense(x, features=50)
        x = nn.tanh(x)
        x = nn.Dense(x, features=50)
        x = nn.tanh(x)
        x = nn.Dense(x, features=50)
        x = nn.tanh(x)
        x = nn.Dense(x, features=1)
        return x

class Model(object):
    """The model including functions named train and loss
    """

    def __init__(self, data, learning_rate):
        self.data = data
        # self.net = net
        self.learning_rate = learning_rate

    def train(
        self,
        epochs=None,
        batch_size=None,
        model_save_path=None,
        display_every=1000,
    ):
        """ Trains the model for a fixed number of epochs"""
        dim_x = self.data.geom.dim
        train_data = self.data.train_data()
        train_points = device_put(train_data[:,dim_x])
        train_tag = device_put(train_data[:,dim_x:])
        print('+-+-+-+-+-+-+-')

        _, initial_params = FNN.init_by_shape(
            jax.random.PRNGKey(0),
            [((1,1,3), jnp.float32)]
        )
        model = nn.Model(FNN, initial_params)

        optimizer_def = flax.optim.Adam(learning_rate = self.learning_rate)
        optimizer = optimizer_def.create(model)
        print('+++++++++++++')

        first_grad = grad(optimizer.target)(train_points)
        second_grad = jax.hessian(optimizer.target)(train_points).diagonal()

        print('------------')
        print(first_grad,second_grad)
        return first_grad, second_grad
    
    @jax.jit
    def train_step(optimizer, points, tag):

        first_grad = grad(optimizer.target)(points)
        second_grad = jax.hessian(optimizer.target)(points).diagonal()
        
        return first_grad, second_grad


