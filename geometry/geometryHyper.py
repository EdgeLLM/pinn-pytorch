from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from SALib.sample import sobol_sequence
from scipy import stats
from sklearn import preprocessing

from .geometry import Geometry


class Hypercube(Geometry):
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")
        if np.any(np.array(xmin) >= np.array(xmax)):
            raise ValueError("xmin >= xmax")

        self.xmin, self.xmax = np.array(xmin), np.array(xmax)
        super(Hypercube, self).__init__(
            len(xmin), (self.xmin, self.xmax), np.linalg.norm(self.xmax - self.xmin)
        )
    def inside(self, x):
        return np.all(x >= self.xmin) and np.all(x <= self.xmax)

    def on_boundary(self, x):
        return self.inside(x) and (
            np.any(np.isclose(x, self.xmin)) or np.any(np.isclose(x, self.xmax))
        )

    def boundary_normal(self, x):
        n = np.zeros(self.dim)
        for i, xi in enumerate(x):
            if np.isclose(xi, self.xmin[i]):
                n[i] = -1
                break
            if np.isclose(xi, self.xmax[i]):
                n[i] = 1
                break
        return n

    def uniform_points(self, n, boundary=False):
        n1 = int(np.ceil(n ** (1 / self.dim)))
        xi = []
        for i in range(self.dim):
            if boundary:
                xi.append(np.linspace(self.xmin[i], self.xmax[i], num=n1))
            else:
                xi.append(
                    np.linspace(self.xmin[i], self.xmax[i], num=n1 + 1, endpoint=False)[
                        1:
                    ]
                )
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_points(self, n, random="pseudo"):
        if random == "pseudo":
            x = np.random.rand(n, self.dim)
        elif random == "sobol":
            x = sobol_sequence.sample(n + 1, self.dim)[1:]
        return (self.xmax - self.xmin) * x + self.xmin

    def periodic_point(self, x, component):
        y = np.copy(x)
        if np.isclose(y[component], self.xmin[component]):
            y[component] += self.xmax[component] - self.xmin[component]
        elif np.isclose(y[component], self.xmax[component]):
            y[component] -= self.xmax[component] - self.xmin[component]
        return y
    