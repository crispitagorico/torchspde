import pytest
import torch
import numpy as np
from torchspde.neural_spde import NeuralSPDE


def test_fixed_point_solver_1d():
    batch, in_channels, dim_x, dim_t,  = 2, 1, 32, 64
    u0 = torch.rand(batch, 1, dim_x, dtype=torch.float32)
    xi = torch.rand(batch, 1, dim_x, dim_t, dtype=torch.float32)
    model = NeuralSPDE(dim=1, in_channels=1, noise_channels=1, hidden_channels=16, n_iter=4, modes1=16, modes2=50).cuda()
    out= model(u0.cuda(), xi.cuda())
    assert out.shape == (batch, in_channels, dim_x, dim_t)

def test_fixed_point_solver_2d():
    batch, in_channels, dim_x, dim_y,  dim_t,  = 2, 1, 16, 16, 32
    u0 = torch.rand(batch, 1, dim_x, dim_y, dtype=torch.float32)
    xi = torch.rand(batch, 1, dim_x, dim_y, dim_t, dtype=torch.float32)
    model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=16, n_iter=4, modes1=16, modes2=16, modes3=20).cuda()
    out= model(u0.cuda(), xi.cuda())
    assert out.shape == (batch, in_channels, dim_x, dim_y, dim_t)


def test_diffeq_solver_1d():
    batch, in_channels, dim_x, dim_t,  = 2, 1, 32, 64
    u0 = torch.rand(batch, 1, dim_x, dtype=torch.float32)
    xi = torch.rand(batch, 1, dim_x, dim_t, dtype=torch.float32)
    model = NeuralSPDE(dim=1, in_channels=1, noise_channels=1, hidden_channels=16, n_iter=4, modes1=16, solver='diffeq').cuda()
    out= model(u0.cuda(), xi.cuda())
    assert out.shape == (batch, in_channels, dim_x, dim_t)

def test_diffeq_solver_2d():
    batch, in_channels, dim_x, dim_y,  dim_t,  = 2, 1, 16, 16, 32
    u0 = torch.rand(batch, 1, dim_x, dim_y, dtype=torch.float32)
    xi = torch.rand(batch, 1, dim_x, dim_y, dim_t, dtype=torch.float32)
    model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=16, n_iter=4, modes1=16, modes2=16, solver='diffeq').cuda()
    out= model(u0.cuda(), xi.cuda())
    assert out.shape == (batch, in_channels, dim_x, dim_y, dim_t)