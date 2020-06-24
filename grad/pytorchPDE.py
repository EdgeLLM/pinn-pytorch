from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import torch
import numpy as np
from torch.autograd.functional import vjp, vhp, jacobian, hessian
from .pytorchGrad import PytorchGrad  
import torch.optim as optim
    
class PytorchPDE(PytorchGrad):
    def __init__(self, model, data, pde, device='cpu', optim_type='adam',lr=0.001, first_grads=True, second_grads=True):
        super(PytorchPDE, self).__init__(model, data, device)
        self.pde = pde
        self.data = data
        self.first_grads_flag = first_grads
        self.second_grads_flag = second_grads
        
        if(optim_type == 'adam'):
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def loss(self):
        
        predicts = self.predict()
        
        if(self.first_grads_flag):
            first_grads = self.first_grads()
        else:
            first_grads = None
            
        if(self.second_grads_flag):
            second_grads = self.second_grads()
        else:
            second_grads = None
            
        domain_loss = self.pde(self.train_points, predicts, first_grads, second_grads)*self.train_tag[:,-1]
        boundary_loss = 0
        for i, bc in enumerate(self.data.bcs):
            if(bc.type=='Dirichlet'):
                bc_loss = bc.error(self.train_points, predicts)*self.train_tag[:,bc.tag]
            elif(bc.type=='Neumann'):
                bc_loss = bc.error(self.train_points, first_grads)*self.train_tag[:,bc.tag]
                
            boundary_loss += torch.sum(bc_loss**2)/torch.sum(self.train_tag[:,bc.tag])
            
            
        return torch.sum(domain_loss**2)/torch.sum(self.train_tag[:,-1]) + boundary_loss
    
    def update_dataset(self):
        self.train_points, self.train_tag = self.data.train_points()
        self.train_points  = torch.from_numpy(self.train_points).to(self.device).float()
        self.train_tag = torch.from_numpy(self.train_tag).to(self.device).float()
        
    def train(self, epochs=5000):
        

        for i in range(epochs):  # loop over the dataset multiple times

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize

            loss = self.loss()
            loss.backward()
            self.optimizer.step()

            # print statistics
            if i % 1000 == 999:    # print every 2000 mini-batches
                self.update_dataset()
                print('i:',i+1,'; test_loss: ',self.loss())
#                 if i % 100000 == 99999:
#                     torch.save(my_nn.state_dict(), '0623_torch_waveguide_1.0z_domain_2e3_bc_3e3_epoch_{}.pth'.format(i+1))

        print('Finished Training')

    

    