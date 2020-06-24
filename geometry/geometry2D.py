from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import itertools
from SALib.sample import sobol_sequence
from scipy import spatial

from .geometry import Geometry
from .geometryHyper import Hypercube


class Rectangle(Hypercube):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        super(Rectangle, self).__init__(xmin, xmax)
        self.perimeter = 2 * np.sum(self.xmax - self.xmin)
        self.area = np.prod(self.xmax - self.xmin)

    def uniform_boundary_points(self, n):
        nx, ny = np.ceil(n / self.perimeter * (self.xmax - self.xmin)).astype(int)
        xbot = np.hstack(
            (
                np.linspace(self.xmin[0], self.xmax[0], num=nx, endpoint=False)[
                    :, None
                ],
                np.full([nx, 1], self.xmin[1]),
            )
        )
        yrig = np.hstack(
            (
                np.full([ny, 1], self.xmax[0]),
                np.linspace(self.xmin[1], self.xmax[1], num=ny, endpoint=False)[
                    :, None
                ],
            )
        )
        xtop = np.hstack(
            (
                np.linspace(self.xmin[0], self.xmax[0], num=nx + 1)[1:, None],
                np.full([nx, 1], self.xmax[1]),
            )
        )
        ylef = np.hstack(
            (
                np.full([ny, 1], self.xmin[0]),
                np.linspace(self.xmin[1], self.xmax[1], num=ny + 1)[1:, None],
            )
        )
        x = np.vstack((xbot, yrig, xtop, ylef))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        x_corner = np.vstack(
            (
                self.xmin,
                [self.xmax[0], self.xmin[1]],
                self.xmax,
                [self.xmin[0], self.xmax[1]],
            )
        )
        n -= 4
        if n <= 0:
            return x_corner

        l1 = self.xmax[0] - self.xmin[0]
        l2 = l1 + self.xmax[1] - self.xmin[1]
        l3 = l2 + l1
        if random == "sobol":
            u = np.ravel(sobol_sequence.sample(n + 4, 1))[2:]
            u = u[np.logical_not(np.isclose(u, l1 / self.perimeter))]
            u = u[np.logical_not(np.isclose(u, l3 / self.perimeter))]
            u = u[:n]
        else:
            u = np.random.rand(n)
        u *= self.perimeter
        x = []
        for l in u:
            if l < l1:
                x.append([self.xmin[0] + l, self.xmin[1]])
            elif l < l2:
                x.append([self.xmax[0], self.xmin[1] + l - l1])
            elif l < l3:
                x.append([self.xmax[0] - l + l2, self.xmax[1]])
            else:
                x.append([self.xmin[0], self.xmax[1] - l + l3])
        return np.vstack((x_corner, x))

    def test_points(self, step):
        xy = np.mgrid[self.xmin[0] : self.xmax[0] + step/10 : step, self.xmin[1] : self.xmax[1] + step/10 : step].reshape(2,-1).T

        x_len = len(np.unique(xy[:,0]))
        y_len = len(np.unique(xy[:,1]))

        return {'x,y': xy}, [x_len, y_len]

class Cuboid(Hypercube):
    """
    Args:
        xmin: Coordinate of bottom left corner.
        xmax: Coordinate of top right corner.
    """

    def __init__(self, xmin, xmax):
        super(Cuboid, self).__init__(xmin, xmax)
        dx = self.xmax - self.xmin
        self.area = 2 * np.sum(dx * np.roll(dx, 2))

    def random_boundary_points(self, n, random="pseudo"):
        x_corner = np.vstack(
            (
                self.xmin,
                [self.xmin[0], self.xmax[1], self.xmin[2]],
                [self.xmax[0], self.xmax[1], self.xmin[2]],
                [self.xmax[0], self.xmin[1], self.xmin[2]],
                self.xmax,
                [self.xmin[0], self.xmax[1], self.xmax[2]],
                [self.xmin[0], self.xmin[1], self.xmax[2]],
                [self.xmax[0], self.xmin[1], self.xmax[2]],
            )
        )
        n -= 8
        if n <= 0:
            return x_corner

        pts = [x_corner]
        density = n / self.area
        rect = Rectangle(self.xmin[:-1], self.xmax[:-1])
        for z in [self.xmin[-1], self.xmax[-1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u, np.full((len(u), 1), z))))
        rect = Rectangle(self.xmin[::2], self.xmax[::2])
        for y in [self.xmin[1], self.xmax[1]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), y), u[:, 1:])))
        rect = Rectangle(self.xmin[1:], self.xmax[1:])
        for x in [self.xmin[0], self.xmax[0]]:
            u = rect.random_points(int(np.ceil(density * rect.area)), random=random)
            pts.append(np.hstack((np.full((len(u), 1), x), u)))
        return np.vstack(pts)

    def uniform_boundary_points(self, n):
        h = (self.area / n) ** 0.5
        nx, ny, nz = np.ceil((self.xmax - self.xmin) / h).astype(int) + 1
        x = np.linspace(self.xmin[0], self.xmax[0], num=nx)
        y = np.linspace(self.xmin[1], self.xmax[1], num=ny)
        z = np.linspace(self.xmin[2], self.xmax[2], num=nz)

        pts = []
        for v in [self.xmin[-1], self.xmax[-1]]:
            u = list(itertools.product(x, y))
            pts.append(np.hstack((u, np.full((len(u), 1), v))))
        if nz > 2:
            for v in [self.xmin[1], self.xmax[1]]:
                u = np.array(list(itertools.product(x, z[1:-1])))
                pts.append(np.hstack((u[:, 0:1], np.full((len(u), 1), v), u[:, 1:])))
        if ny > 2 and nz > 2:
            for v in [self.xmin[0], self.xmax[0]]:
                u = list(itertools.product(y[1:-1], z[1:-1]))
                pts.append(np.hstack((np.full((len(u), 1), v), u)))
        pts = np.vstack(pts)
        if n != len(pts):
            print(
                "Warning: {} points required, but {} points sampled.".format(
                    n, len(pts)
                )
            )
        return pts
    
    def test_points(self, step):
        X,Y = np.mgrid[self.xmin[0] : self.xmax[0] + step/10 : step, self.xmin[1] : self.xmax[1] + step/10 : step]

        xy = np.vstack(X.flatten(), Y.flatten()).T

        x_len = len(np.unique(X))
        y_len = len(np.unique(Y))

        return {'x,y': xy}, [x_len, y_len]
