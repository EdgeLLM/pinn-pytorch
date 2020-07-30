from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

class BC(object):
    """Boundary conditions.

    Args:
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.
        component: The output component satisfying this BC.
    """

    def __init__(self, geom, on_boundary, component_x, component_y):
        self.geom = geom
        self.on_boundary = on_boundary
        self.component_x = component_x
        self.component_y = component_y 
        self.inputdim = self.geom.dim
        self.outputdim = None
        self.tag = None

    def set_tag(self, tag):
        self.tag = tag
        
    def filter(self, X):
        X = np.array([x for x in X if self.on_boundary(x, self.geom.on_boundary(x))])
        return X if len(X) > 0 else np.empty((0, self.geom.dim))

    def collocation_points(self, X):
        return self.filter(X)

    def error(self, X, inputs, outputs, beg, end):
        raise NotImplementedError(
            "{}.error to be implemented".format(type(self).__name__)
        )


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x).
    """

    def __init__(self, geom, func, on_boundary, component_x=0, component_y=0):
        super(DirichletBC, self).__init__(geom, on_boundary, component_x, component_y)
        self.func = func
        self.type = 'Dirichlet'

    def error(self, X, predicts):
        values = torch.squeeze(self.func(X))
        if len(values.shape) != 1:
            raise RuntimeError(
                "DirichletBC should output 1D values. Use argument 'component' for different components."
            )

        return predicts[:,self.component_y] - values


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x).
    """

    def __init__(self, geom, func, on_boundary, component_x=0, component_y=0):
        super(NeumannBC, self).__init__(geom, on_boundary, component_x, component_y)
        self.func = func
        self.type = 'Neumann'

    def error(self, X, first_grads):
        column_index = self.component_y * self.inputdim + self.component_x
        return first_grads[:, column_index] - torch.squeeze(self.func(X)).cuda()

class HessianBC(BC):
    """Hessian boundary conditions: d^2y/d(x1^2) = func(x) or d^2y/d(x1 x2).
    """

    def __init__(self, geom, func, on_boundary, component_x=[], component_y=0):
        super(HessianBC, self).__init__(geom, on_boundary, component_x, component_y)
        self.func = func
        self.type = 'Hessian'

    def error(self, X, second_grads):
        
        if(len(self.component_x) != 2):
            raise RuntimeError(
                "HessianBC is used for second order grads."
            )
            
        column_index = self.inputdim * self.component_x[0] + self.component_x[1]
        
        return second_grads[self.component_y][:, column_index] - torch.squeeze(self.func(X)).cuda()