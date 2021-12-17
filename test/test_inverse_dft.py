import pytest
import torch
import numpy as np
from torchspde.solver_picard import inverseDFTn
from torchspde.neural_spde import NeuralSPDE

@pytest.mark.parametrize("dim_phys, modes, dim_fft", (([4, 8], None, [2]),
                                                      ([4, 8], None, [3]),
                                                      ([4, 8], None, [2, 3]),
                                                      ([14, 20], [8, 20], [2]),
                                                      ([20, 14], [20, 8], [3]),
                                                      ([14, 14], [8, 6],  [2, 3])))
def test_inverseDFT1D(dim_phys, modes, dim_fft):
    
    # create space-time grid of points 
    dim_x, dim_t = dim_phys[0], dim_phys[1]
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float).reshape(1, dim_t).repeat(dim_x, 1)
    gridx = torch.tensor(np.linspace(0, 1, dim_x+1)[:-1], dtype=torch.float).reshape(dim_x, 1).repeat(1, dim_t)
    if dim_fft == [2, 3]:
        grid = torch.stack([gridx, gridt], dim=-1)
    elif dim_fft == [2]:
        grid = gridx[:, 0].unsqueeze(-1)
    elif dim_fft == [3]:
        grid = gridt[0, :].unsqueeze(-1)

    # determine whether we want to recover more points
    if modes is None:
        modes = dim_phys
        s = None
    else:
        s = [dim_phys[i-2] for i in dim_fft]

    # create tensor in spectral domain
    batch, channels = 2, 4
    u_ft = torch.rand(batch, channels, modes[0], modes[1], dtype=torch.complex64)

    # compute the inverse DFT via our implementation and Pytorch's
    ans_torch = torch.fft.ifftn(u_ft, dim=dim_fft, s=s)
    ans_ours = inverseDFTn(u_ft, grid, dim=dim_fft, s=s)

    # check answers
    torch.testing.assert_allclose(ans_torch, ans_ours, rtol=1e-03, atol=1e-08)

@pytest.mark.parametrize("dim_phys, modes, dim_fft", (([4, 6, 8], None, [2, 3]),
                                                      ([4, 6, 8], None, [4]),
                                                      ([4, 6, 8], None, [2, 3, 4]),
                                                      ([14, 16, 20], [8, 6, 20], [2, 3]),
                                                      ([20, 16, 14], [20, 16, 8], [4]),
                                                      ([14, 12, 14], [8, 6, 6],  [2, 3, 4])))
def test_inverseDFT2D(dim_phys, modes, dim_fft):
    
    # create space-time grid of points 
    dim_x, dim_y, dim_t = dim_phys[0], dim_phys[1], dim_phys[2]
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float).reshape(1, 1, dim_t).repeat(dim_x, dim_y, 1)
    gridx = torch.tensor(np.linspace(0, 1, dim_x+1)[:-1], dtype=torch.float).reshape(dim_x, 1, 1).repeat(1, dim_y, dim_t)
    gridy = torch.tensor(np.linspace(0, 1, dim_y+1)[:-1], dtype=torch.float).reshape(1, dim_y, 1).repeat(dim_x, 1, dim_t)
    if dim_fft == [2, 3, 4]:
        grid = torch.stack([gridx, gridy, gridt], dim=-1)
    elif dim_fft == [2, 3]:
        grid = torch.stack([gridx[...,0], gridy[...,0]], dim=-1)
    elif dim_fft == [4]:
        grid = gridt[0,0,:].unsqueeze(-1)

    # determine whether we want to recover more points
    if modes is None:
        modes = dim_phys
        s = None
    else:
        s = [dim_phys[i-2] for i in dim_fft]
    
    # create tensor in spectral domain
    batch, channels = 2, 4
    u_ft = torch.rand(batch, channels, modes[0], modes[1], modes[2], dtype=torch.complex64)

    # compute the inverse DFT via our implementation and Pytorch's
    ans_torch = torch.fft.ifftn(u_ft, dim=dim_fft, s=s)
    ans_ours = inverseDFTn(u_ft, grid, dim=dim_fft, s=s)

    # check answers
    torch.testing.assert_allclose(ans_torch, ans_ours, rtol=1e-03, atol=1e-08)


def test_NSPDE_inverseDFT1D():

    batch, dim_x, dim_t = 2, 32, 64
    u0 = torch.rand(batch, 1, dim_x, dtype=torch.float32)
    xi = torch.rand(batch, 1, dim_x, dim_t, dtype=torch.float32)

    # create space-time grid of points 
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float32).reshape(1, 1, dim_t).repeat(batch, dim_x, 1)
    gridx = torch.tensor(np.linspace(0, 1, dim_x+1)[:-1], dtype=torch.float32).reshape(1, dim_x, 1).repeat(batch, 1, dim_t)
    grid = torch.stack([gridx, gridt], dim=-1)

    model = NeuralSPDE(dim=1, in_channels=1, noise_channels=1, hidden_channels=16, n_iter=4, modes1=16, modes2=50).cuda()

    out_physicsinformed = model(u0.cuda(), xi.cuda(), grid.cuda())
    out = model(u0.cuda(), xi.cuda())

    torch.testing.assert_allclose(out_physicsinformed, out, rtol=1e-03, atol=1e-08)



def test_NSPDE_inverseDFT2D():
    batch, dim_x, dim_y,  dim_t = 2, 16, 16, 10
    u0 = torch.rand(batch, 1, dim_x, dim_y, dtype=torch.float32)
    xi = torch.rand(batch, 1, dim_x, dim_y, dim_t, dtype=torch.float32)

    # create space-time grid of points 
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float32).reshape(1, 1, 1, dim_t).repeat(batch, dim_x, dim_y, 1)
    gridx = torch.tensor(np.linspace(0, 1, dim_x+1)[:-1], dtype=torch.float32).reshape(1, dim_x, 1, 1).repeat(batch, 1, dim_y, dim_t)
    gridy = torch.tensor(np.linspace(0, 1, dim_y+1)[:-1], dtype=torch.float32).reshape(1, 1, dim_y, 1).repeat(batch, dim_x, 1, dim_t)
    grid = torch.stack([gridx, gridy, gridt], dim=-1)

    model = NeuralSPDE(dim=2, in_channels=1, noise_channels=1, hidden_channels=8, n_iter=4, modes1=8, modes2=8, modes3=8).cuda()
    
    out_physicsinformed = model(u0.cuda(), xi.cuda(), grid.cuda())
    out = model(u0.cuda(), xi.cuda())

    torch.testing.assert_allclose(out_physicsinformed, out, rtol=1e-03, atol=1e-08)


