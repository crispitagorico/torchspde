import torch
import numpy as np
from .neural_spde import NeuralSPDE
from torch.autograd import grad

def grad_var(u_i, grid_var):
    """ Input:
              - u_i (batch, dim_x, dim_y, dim_t)
              - grid_var (batch, dim_x, dim_y, dim_t)
        Returns:
              - du_i/dvar(grid_var)  (batch, dim_x, dim_y, dim_t)
    """
    return grad(u_i.sum(), grid_var, create_graph=True)[0] 


def grad_space(u_i, gridx, gridy):
    """ Input:
              - u_i (batch, dim_x, dim_y, dim_t)
              - gridx, gridy (batch, dim_x, dim_y, dim_t)
        Returns:
              - (du_i/dx, du_i/dy)  (batch, dim_x, dim_y, dim_t, 2)
    """
    return torch.stack([grad_var(u_i, gridx), grad_var(u_i, gridy)], dim=-1)


def grad_space_perp(u_i, gridx, gridy):
    """ Input:
              - u_i (batch, dim_x, dim_y, dim_t)
              - gridx, gridy (batch, dim_x, dim_y, dim_t)
        Returns:
              - (-du_i/dy, du_i/dx)  (batch, dim_x, dim_y, dim_t, 2)
    """
    return torch.stack([-grad_var(u_i, gridy), grad_var(u_i, gridx)], dim=-1)


def laplacian(u_i, gridx, gridy):
    """ Input:
              - u_i (batch, dim_x, dim_y, dim_t)
              - gridx, gridy (batch, dim_x, dim_y, dim_t)
        Returns:
              - d^2u_i/dx^2 + d^2u_i/dy_2  (batch, dim_x, dim_y, dim_t)
    """
    return grad_var(grad_var(u_i, gridx), gridx) + grad_var(grad_var(u_i, gridy), gridy)





#==========================
# Example
#==========================

#batch, in_channels, noise_channels, dim_x, dim_y, dim_t = 5, 3, 2, 8, 8, 10
#u0 = torch.rand(batch, in_channels, dim_x, dim_y, dtype=torch.float32)
#xi = torch.rand(batch, noise_channels, dim_x, dim_y, dim_t, dtype=torch.float32)

#gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float32, requires_grad=True).reshape(1, 1, 1, dim_t).repeat(batch, dim_x, dim_y, 1)
#gridx = torch.tensor(np.linspace(0, 1, dim_x+1)[:-1], dtype=torch.float32, requires_grad=True).reshape(1, dim_x, 1, 1).repeat(batch, 1, dim_y, dim_t)
#gridy = torch.tensor(np.linspace(0, 1, dim_y+1)[:-1], dtype=torch.float32, requires_grad=True).reshape(1, 1, dim_y, 1).repeat(batch, dim_x, 1, dim_t)
#grid = torch.stack([gridx, gridy, gridt], dim=-1)

#model = NeuralSPDE(dim=2, in_channels=in_channels, noise_channels=noise_channels, hidden_channels=16, n_iter=4, modes1=8, modes2=8, modes3=6).cuda()
#out = model(u0.cuda(), xi.cuda(), grid.cuda())

#D_t = grad_var(out[:,0,...], gridt)
#Lap = laplacian(out[:,0,...], gridx, gridy)