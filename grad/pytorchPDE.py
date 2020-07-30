from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import torch
import numpy as np
from torch.autograd.functional import vjp, vhp, jacobian, hessian
from .pytorchGrad import PytorchGrad  
import torch.optim as optim

import neptune
    
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
        
        train_points = self.train_points.clone().detach().requires_grad_(True)
        predicts = self.predict(train_points)
#         origin_predict = self.origin_predict()
#         print('train_poins_min: ', train_points.min())
#         print('train_point_max: ', train_points.max())
        
        self.outshape = predicts.shape
        
        if(self.first_grads_flag):
#             first_grads = self.first_grads()
            first_grads = self.mvjp(train_points)
#             origin_first_grads = self.first_grads()
        else:
            first_grads = None
            
        if(self.second_grads_flag):
#             second_grads = self.second_grads()
            second_grads = self.mvhp(train_points)
#             origin_second_grads = self.second_grads()
        else:
            second_grads = None
            
#         print('train_tag_min: ', self.train_tag[:,-1].min())
#         print('train_tag_max: ', self.train_tag[:,-1].max())
#         print('predicts_min: ', predicts.min())
#         print('first_grads_min: ', first_grads.min())
#         print('second_grads_min: ', second_grads[0].min(), second_grads[0].min())
        
        domain_loss = self.pde(train_points, predicts, first_grads, second_grads)*self.train_tag[:,-1]
        boundary_loss = 0
        for i, bc in enumerate(self.data.bcs):

            if(bc.type=='Dirichlet'):
                bc_loss = bc.error(train_points, predicts)*self.train_tag[:,bc.tag]
            elif(bc.type=='Neumann'):
                bc_loss = bc.error(train_points, first_grads)*self.train_tag[:,bc.tag]
            elif(bc.type=='Hessian'):
                bc_loss = bc.error(train_points, second_grads)*self.train_tag[:,bc.tag]
                
            boundary_loss += torch.sum(bc_loss**2)/torch.sum(self.train_tag[:,bc.tag])
            
#         print('domain_loss: ', domain_loss)
#         print('domain_loss_min: ', domain_loss.min())
#         print('domain_loss_max: ', domain_loss.max())
#         print('sum_domain_loss: ',torch.sum(domain_loss**2))
#         print('sum_domain_flag: ',torch.sum(self.train_tag[:,-1]))
#         print(torch.sum(domain_loss**2)/torch.sum(self.train_tag[:,-1]) + boundary_loss)


        weights_loss = 0
        num_activation_layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if('alpha' in name):
                    num_activation_layer = num_activation_layer + 1
                    weights_loss = weights_loss + torch.exp(param)
                    
        if num_activation_layer > 0:
            weights_loss = weights_loss / num_activation_layer
            weights_loss = 1/weights_loss
        
        return torch.sum(domain_loss)/torch.sum(self.train_tag[:,-1]) + boundary_loss + weights_loss
    
    def update_dataset(self):
        self.train_points, self.train_tag = self.data.train_points()
        self.train_points  = torch.from_numpy(self.train_points).to(self.device).float()
        self.train_tag = torch.from_numpy(self.train_tag).to(self.device).float()
        
    def train(self, logname, epochs=5000, save_step = 10000, update_step=1000, upload=True):
        
        if(upload):
            neptune.init('xiaolinhu/pinnade')
            neptune.create_experiment(name=logname)

        # log some metrics
        
        for i in range(epochs):  # loop over the dataset multiple times

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize

            loss = self.loss()
            
            if(upload):
                neptune.log_metric('loss', loss.item())
            
            loss.backward()
            self.optimizer.step()

            # print statistics
            if i % update_step == update_step-1:    # print every 2000 mini-batches
                self.update_dataset()
                print('i:',i+1,'; test_loss: ',loss.item())
                if i % save_step == save_step-1:
                    torch.save(self.model.state_dict(), logname + '_epoch_{}.pth'.format(i+1))

        print('Finished Training')

    

    