from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Data(object):
    """ Dataset of PDE solver """
    def __init__(
        self,
        geom,
        bcs,
        num_domain=0,
        num_boundary=0,
        train_distribution="random",
        num_test=None,
    ):
        self.geom = geom
        self.bcs = bcs if isinstance(bcs, (list, tuple)) else [bcs]
        self.num_boundary = num_boundary
        self.num_domain = num_domain
        self.train_distribution = train_distribution
        self.num_test = num_test
    
    def get_points(self):
        domain_point = np.empty((0, self.geom.dim))
        boundary_point = np.empty((0, self.geom.dim))
        if self.num_domain > 0:
            if self.train_distribution == "uniform":
                domain_point = self.geom.uniform_points(self.num_domain, boundary=False)
            else:
                domain_point = self.geom.random_points(self.num_domain, random="pseudo")
        if self.num_boundary > 0:
            if self.train_distribution == "uniform":
                boundary_point = self.geom.uniform_boundary_points(self.num_boundary)
            else:
                boundary_point = self.geom.random_boundary_points(self.num_boundary, random="pseudo")
        return domain_point, boundary_point

    def bc_points(self, boundary_point):
        num_bcs = len(self.bcs)
        x_bcs = np.empty((0, self.geom.dim + num_bcs + 1))
        
        for i, bc in enumerate(self.bcs):
            bc_points = bc.collocation_points(boundary_point)
            bc_tag = np.zeros((len(bc_points), num_bcs + 1))
            bc_tag[:,i] = 1
            bc_points_tag = np.hstack((bc_points, bc_tag))
            # print("bc_points_tag: ", bc_points_tag)
            x_bcs = np.vstack((x_bcs, bc_points_tag))
            bc.set_tag(i)

        return x_bcs

    def train_points(self):
        """Prepare training data and training tag.
        """
        
        domain_point, boundary_point = self.get_points()
        bc_points_tag = self.bc_points(boundary_point)
        domain_tag = np.zeros((len(domain_point), len(self.bcs) + 1))
        domain_tag[:,-1] = 1
        domain_points_tag = np.hstack((domain_point, domain_tag))
        
        datasets = np.vstack((domain_points_tag, bc_points_tag))
        return datasets[:,:self.geom.dim], datasets[:,self.geom.dim:]
        
    def test_points(self, step):
        return self.geom.test_points(step)

