from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import itertools
from SALib.sample import sobol_sequence
from scipy import spatial

from .geometry import Geometry
from .geometryHyper import Hypercube
from .geometry2D import Rectangle

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

    def test_points(self, step, type='2D'):

        xy = np.mgrid[self.xmin[0] : self.xmax[0] + step/10 : step, self.xmin[1] : self.xmax[1] + step/10 : step].reshape(2,-1).T

        xz = np.mgrid[self.xmin[0] : self.xmax[0] + step/10 : step, self.xmin[2] : self.xmax[2] + step/10 : step].reshape(2,-1).T

        yz = np.mgrid[self.xmin[1] : self.xmax[1] + step/10 : step, self.xmin[2] : self.xmax[2] + step/10 : step].reshape(2,-1).T

        # X,Y,Z = np.mgrid[self.xmin[0] : self.xmax[0] + step/10 : step, self.xmin[1] : self.xmax[1] + step/10 : step, self.xmin[2] : self.xmax[2] + step/10 : step]

        xy_z_min = np.hstack((xy, np.full((xy.shape[0],1),self.xmin[2])))
        xy_z_max = np.hstack((xy, np.full((xy.shape[0],1),self.xmax[2])))

        xz_y_min = np.insert(xz, [1], np.full((xz.shape[0],1),self.xmin[1]), axis=1)
        xz_y_max = np.insert(xz, [1], np.full((xz.shape[0],1),self.xmax[1]), axis=1)

        yz_x_min = np.hstack((np.full((yz.shape[0],1),self.xmin[0]), yz))
        yz_x_max = np.hstack((np.full((yz.shape[0],1),self.xmax[0]), yz))

        x_len = len(np.unique(xy_z_min[:,0]))
        y_len = len(np.unique(xy_z_min[:,1]))
        z_len = len(np.unique(xz_y_min[:,2]))

        return {'z_min': xy_z_min,
                'z_max': xy_z_max,
                'y_min': xz_y_min,
                'y_max': xz_y_max,
                'x_min': yz_x_min,
                'x_max': yz_x_max}, [x_len, y_len, z_len]