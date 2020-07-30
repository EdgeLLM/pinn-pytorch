from functools import partial # for use with vmap

import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_2D(net_apply, data, step=0.001, model_path=None):

    plot_dict = dict_2D(net_apply, data, step, model_path)

    fig, axs = plt.subplots(figsize=(10, 4))

    i = 0
    
    for key,value in plot_dict.items():
        
        if key == 'x,y':
            key_trans = 'u(x,y)'
            extent_x_min = data.geom.xmin[0]
            extent_x_max = data.geom.xmax[0]
            extent_y_min = data.geom.xmin[1]
            extent_y_max = data.geom.xmax[1]
            x_label = 'x'
            y_label = 'y'
           
        else:
            raise ValueError("The key is not right.")
        
        im = axs.imshow(value.detach().numpy(),extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=-1, vmax=1,  interpolation='nearest', cmap='seismic', origin='lower', aspect='auto')
        axs.set_title(key_trans, fontsize=20)
        axs.set_xlabel(x_label, fontsize=20) 
        axs.set_ylabel(y_label, fontsize=20) 
        axs.tick_params(labelsize='large')
        fig.colorbar(im, ax=axs)
        i = i + 1
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()

def dict_2D(net_apply, data, step, model_path):
    if model_path != None:
        net_apply.load_state_dict(torch.load(model_path))
        net_apply.eval()
        
    net_apply.to('cpu')
    test_dict, test_len = data.test_points(step)
    plot_dict = {}
    for key,value in test_dict.items():
        test_input = torch.from_numpy(value).float()
        test_predict = net_apply(test_input)
        
        if key == 'x,y':
            plot_array = (test_predict.reshape(test_len[0],test_len[1])).T

        else:
            raise ValueError("The key is not right.")
        
        plot_dict[key] = plot_array
    return plot_dict

def torch_plot_3D_boundary(model_path, net_apply, data, plat='jax',step=0.001, component_y=0):

    plot_dict = dict_3D_bondary(model_path, net_apply, data, step, plat, component_y)

    fig, axs = plt.subplots(2,3, figsize=(15, 10))
    # fig, axs = plt.subplots(2,3)
    axs = axs.ravel()

    i = 0
    
    for key,value in plot_dict.items():
        
        if key == 'z_min':
            key_trans = 'Hz(x,y,%g)' % data.geom.xmin[2]
            extent_x_min = data.geom.xmin[0]
            extent_x_max = data.geom.xmax[0]
            extent_y_min = data.geom.xmin[1]
            extent_y_max = data.geom.xmax[1]
            x_label = 'x'
            y_label = 'y'

        elif key == 'z_max':
            key_trans = 'Hz(x,y,%g)' % data.geom.xmax[2]
            extent_x_min = data.geom.xmin[0]
            extent_x_max = data.geom.xmax[0]
            extent_y_min = data.geom.xmin[1]
            extent_y_max = data.geom.xmax[1]
            x_label = 'x'
            y_label = 'y'
        elif key == 'y_min':
            key_trans = 'Hz(x,%g,z)' % data.geom.xmin[1]
            extent_x_min = data.geom.xmin[0]
            extent_x_max = data.geom.xmax[0]
            extent_y_min = data.geom.xmin[2]
            extent_y_max = data.geom.xmax[2]
            x_label = 'x'
            y_label = 'z'

        elif key == 'y_max':
            key_trans = 'Hz(x,%g,z)' % data.geom.xmax[1]
            extent_x_min = data.geom.xmin[0]
            extent_x_max = data.geom.xmax[0]
            extent_y_min = data.geom.xmin[2]
            extent_y_max = data.geom.xmax[2]
            x_label = 'x'
            y_label = 'z'


        elif key == 'x_min':
            key_trans = 'Hz(%g,y,z)' % data.geom.xmin[0]
            extent_x_min = data.geom.xmin[1]
            extent_x_max = data.geom.xmax[1]
            extent_y_min = data.geom.xmin[2]
            extent_y_max = data.geom.xmax[2]
            x_label = 'y'
            y_label = 'z'


        elif key == 'x_max':
            key_trans = 'Hz(%g,y,z)' % data.geom.xmax[0]
            extent_x_min = data.geom.xmin[1]
            extent_x_max = data.geom.xmax[1]
            extent_y_min = data.geom.xmin[2]
            extent_y_max = data.geom.xmax[2]
            x_label = 'y'
            y_label = 'z'

           
        else:
            raise ValueError("The key is not right.")
        
#         im = axs[i].imshow(value,extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), vmin=-1, vmax=1,  interpolation='nearest', cmap='seismic', origin='lower', aspect='auto')
        im = axs[i].imshow(value,extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max), interpolation='nearest', cmap='seismic', origin='lower', aspect='auto')
        axs[i].set_title(key_trans, fontsize=16)
        axs[i].set_xlabel(x_label, fontsize=16.0) 
        axs[i].set_ylabel(y_label, fontsize=16.0) 
        axs[i].tick_params(labelsize='large')

        fig.colorbar(im, ax=axs[i])
        i = i + 1
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()

def dict_3D_bondary(model_path, net_apply, data, step, plat, component_y):
    if plat == 'jax':
        net_params_load = np.load(model_path, allow_pickle=True)
    elif model_path == 'exact':
        print("Output Exact Solution.")
    elif plat == 'torch':
        net_apply.load_state_dict(torch.load(model_path))
        net_apply.eval()
    else:
        raise Exception("No corresponding platform!")
    test_dict, test_len = data.test_points(step)
    plot_dict = {}
    for key,value in test_dict.items():
        if plat == 'jax':
            test_input = device_put(value)
        #     print(test_input)
            test_predict = vmap(partial(net_apply, net_params_load))(test_input)
        elif plat == 'torch':
            with torch.no_grad():
                test_input = torch.from_numpy(value).float()
                test_predict = net_apply(test_input)
        else:
            raise Exception("No corresponding platform!")
    #     test_predict = batched_predict(net_params,test_input)
    #     print(test_predict)
        if key == 'z_min' or key == 'z_max':
            plot_array = (test_predict[:,component_y].reshape(test_len[0],test_len[1])).T
            
        elif key == 'y_min' or key == 'y_max':
            plot_array = (test_predict[:,component_y].reshape(test_len[0],test_len[2])).T
            
        elif key == 'x_min' or key == 'x_max':
            plot_array = (test_predict[:,component_y].reshape(test_len[1],test_len[2])).T
            
        else:
            raise ValueError("The key is not right.")
        
        plot_dict[key] = plot_array
    return plot_dict