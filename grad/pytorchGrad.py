from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import torch
import numpy as np
from torch.autograd.functional import vjp, vhp, jacobian, hessian

class PytorchGrad(object):
    def __init__(self, model, data, device):
        self.model = model
        self.device = device
        self.train_points, self.train_tag = data.train_points()
        self.train_points  = torch.from_numpy(self.train_points).to(self.device).float()
        self.train_tag = torch.from_numpy(self.train_tag).to(self.device).float()
        
        
    def vjp_reducer(self, x):
        return self.model(x).sum(1)
    
    def first_grads(self):
        
        v = torch.ones(self.train_points.shape[0]).to(self.device)
        predict_vjp = vjp(self.vjp_reducer, self.train_points,v, create_graph=True)

        predict_value = predict_vjp[0]
        first_grad = predict_vjp[1]
        return first_grad
    
    def hessian_reducer(self, x):
        return self.model(x).sum()
    
    def predict(self):
        return self.model(self.train_points).sum(1)
    
    
        
    def second_grads(self):
        m,n = self.train_points.shape[0],self.train_points.shape[1]
        second_grad_list = []
        for i in range(n):
            v = torch.cat([torch.ones(m,1)*(i==j) for j in range(n)],1).to(self.device)
            hessian_vhp = vhp(self.hessian_reducer, self.train_points, v, create_graph=True)[1]
            second_grad_list.append(hessian_vhp[:,i])
        second_grad = torch.stack(second_grad_list,0).T
        return second_grad
    