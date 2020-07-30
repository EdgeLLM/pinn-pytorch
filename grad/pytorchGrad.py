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
        self.inshape = self.train_points.shape
        self.outshape = None
        
    def predict(self, train_points):
        return self.model(train_points)
    
    def origin_predict(self):
        return self.model(self.train_points).sum(1)
    
    def vjp_reducer(self, x):
        
        r""" To do.
        Args:
            x : To do.
        Returns:
            To do.
        Example:
            >>> x
            tensor([[0.2513, 0.6246],
                    [0.5098, 0.8833],
                    [0.1971, 0.7218]], device='cuda:0')

            >>> vjp_reducer(x)
            tensor([2.8675, 3.0116, 2.8603], 
                    device='cuda:0', grad_fn=<SumBackward1>)

        """

        return self.model(x).sum(1)
    
    def first_grads(self):
        r""" To do.
        Args:
            x : To do.
        Returns:
            To do.
        Example:
            >>> self.train_points
            tensor([[0.2513, 0.6246],
            [0.5098, 0.8833],
            [0.1971, 0.7218]], device='cuda:0')

            >>> first_grads()
            tensor([[0.3441, 0.1167],
                    [0.4489, 0.2082],
                    [0.3500, 0.1231]], 
                    device='cuda:0', grad_fn=<MmBackward>)

        """
        
        v = torch.ones(self.train_points.shape[0]).to(self.device)
        predict_vjp = vjp(self.vjp_reducer, self.train_points,v, create_graph=True)

        predict_value = predict_vjp[0]
        first_grad = predict_vjp[1]
        return first_grad

    
    def hessian_reducer(self, x):
        return self.model(x).sum()
    
    def second_grads(self):
        m,n = self.outshape[0],self.inshape[1]
        second_grad_list = []
        for i in range(n):
            v = torch.cat([torch.ones(m,1)*(i==j) for j in range(n)],1).to(self.device)
            hessian_vhp = vhp(self.hessian_reducer, self.train_points, v, create_graph=True)[1]
            second_grad_list.append(hessian_vhp[:,i])
        second_grad = torch.stack(second_grad_list,0).T
        return second_grad
    
    def mvjp_reducer(self, x):
        
        r""" To do.
        Args:
            x : To do.
        Returns:
            To do.
        Example:
            >>> x
            tensor([[0.2513, 0.6246],
                    [0.5098, 0.8833],
                    [0.1971, 0.7218]], device='cuda:0')

            >>> vjp_reducer(x)
            tensor([2.8675, 3.0116, 2.8603], 
                    device='cuda:0', grad_fn=<SumBackward1>)

        """

        return self.model(x)
    
    def mvjp(self, train_points):
        r""" vjp for multi outputs.  This function is as fast as first_grads function.
        Args:
            outdim: output shape 
        Returns:
            [[(dy_1)^1/dx_1, (dy_1)^1/dx_2, (dy_1)^2/dx_1, (dy_1)^2/dx_2],
             [(dy_2)^1/dx_1, (dy_2)^1/dx_2, (dy_2)^2/dx_1, (dy_2)^2/dx_2],
             [(dy_3)^1/dx_1, (dy_3)^1/dx_2, (dy_3)^2/dx_1, (dy_3)^2/dx_2]]
        Example:
            >>> self.train_points
            tensor([[0.3191, 0.2468],
                    [0.0719, 0.2555],
                    [0.7084, 0.0403]], device='cuda:0')

            >>> mvjp(outdim)
            tensor([[ 0.2953, -0.2649, -0.0216,  0.4649],
                    [ 0.2564, -0.2301, -0.0218,  0.4693],
                    [ 0.4057, -0.3640, -0.0194,  0.4185]], device='cuda:0', grad_fn=<MmBackward>)

        """
        m,n = self.outshape[0], self.outshape[1]
        first_grad_list = []
        for i in range(n):
            v = torch.cat([torch.ones(m,1)*(i==j) for j in range(n)],1).cuda()
            jacobian_vjp = vjp(self.mvjp_reducer, train_points, v, create_graph=True)[1]
            first_grad_list.append(jacobian_vjp)
        first_grad = torch.cat(first_grad_list,1)
        
        return first_grad
    
    def mvhp(self, train_points):
        r""" vhp for multi outputs. This function is as fast as second_grads function.
        Returns:
            To do;
        Example:
            >>> self.train_points
            tensor([[0.0293, 0.2130],
                    [0.6103, 0.7452],
                    [0.6648, 0.5782]], device='cuda:0')

            >>> mvjp(outdim)
            tensor([[0.5848, 0.2233, 0.2233, 0.0852, 0.0860, 0.1133, 0.1133, 0.1492],
                    [0.3434, 0.1311, 0.1311, 0.0501, 0.0601, 0.0792, 0.0792, 0.1043],
                    [0.3456, 0.1319, 0.1319, 0.0504, 0.0630, 0.0829, 0.0829, 0.1092]],
                   device='cuda:0', grad_fn=<CatBackward>)

        """
        m,n,k = self.outshape[0], self.outshape[1] * self.inshape[1], self.inshape[1] ** 2
        second_grad_list = []
        for i in range(n):
            v = torch.cat([torch.ones(m,1)*(i==j) for j in range(n)],1).cuda()
            hessian_vjp = vjp(self.mvjp, train_points, v, create_graph=True)[1]
            second_grad_list.append(hessian_vjp)
        second_grad = torch.cat(second_grad_list,1)
        second_grad = torch.split(second_grad, k, 1)
        return second_grad
    

    

    