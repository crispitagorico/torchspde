# adapted from https://github.com/zongyi-li/fourier_neural_operator

import torch
import csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import operator
import itertools
from functools import reduce
from functools import partial

from utilities import LpLoss, count_params, EarlyStopping

#===========================================================================
# 2d fourier layers
#===========================================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,t), (in_channel, out_channel, x,t) -> (batch, out_channel, x,t)
        return torch.einsum("bixt,ioxt->boxt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO_layer(nn.Module):
    def __init__(self, modes1, modes2, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last

        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        # self.bn = torch.nn.BatchNorm2d(width)


    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)
            
        return x


class FNO_space1D_time(nn.Module):
    def __init__(self, modes1, modes2, width, L, T):
        super(FNO_space1D_time, self).__init__()

        """
        The overall network. It contains L layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. L layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: a driving function observed at T timesteps + 2 locations (u(1, x), ..., u(T, x),  x, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, t=T, c=T+2)
        output: the solution at T timesteps
        output shape: (batchsize, x=64, t=T, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.L = L
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(T+2, self.width)
        # input channel is T+2: the solution of the first T timesteps + 2 locations (u(1, x), ..., u(T, x),  x, t)

        # self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)
        # self.bn0 = torch.nn.BatchNorm2d(self.width)
        # self.bn1 = torch.nn.BatchNorm2d(self.width)
        # self.bn2 = torch.nn.BatchNorm2d(self.width)
        # self.bn3 = torch.nn.BatchNorm2d(self.width)
        self.net = [ FNO_layer(modes1, modes2, width) for i in range(self.L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.net = nn.Sequential(*self.net)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """ - x: (batch, dim_x, T_out, T)
        """
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        # x1 = self.conv0(x)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2

        x = self.net(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_t = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, size_t, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridt), dim=-1).to(device)


#===========================================================================
# Data Loaders
#===========================================================================

def dataloader_fno_1d_xi(u, xi, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, dataset=None):

    if dataset=='phi41':
        T, sub_t = 51, 1
    elif dataset=='wave':
        T, sub_t = (u.shape[-1]+1)//2, 5

    u_train = u[:ntrain, :dim_x, 0:T:sub_t]
    xi_ = torch.diff(xi[:ntrain, :dim_x, 0:T:sub_t], dim=-1) 
    xi_ = torch.cat([torch.zeros_like(xi_[..., 0].unsqueeze(-1)), xi_], dim=-1)
    xi_train = xi_[:ntrain].reshape(ntrain, dim_x, 1, xi_.shape[-1]).repeat([1, 1, xi_.shape[-1], 1])

    u_test = u[-ntest:, :dim_x, 0:T:sub_t]
    xi_ = torch.diff(xi[-ntest:, :dim_x, 0:T:sub_t], dim=-1) 
    xi_ = torch.cat([torch.zeros_like(xi_[..., 0].unsqueeze(-1)), xi_], dim=-1)
    xi_test = xi_[-ntest:].reshape(ntest, dim_x, 1, xi_.shape[-1]).repeat([1, 1, xi_.shape[-1], 1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def dataloader_fno_1d_u0(u, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, dataset=None):

    if dataset=='phi41':
        T, sub_t = 51, 1
    elif dataset=='wave':
        T, sub_t = (u.shape[-1]+1)//2, 5

    u_train = u[:ntrain, :-1, 0:T:sub_t]
    u0_train = u[:ntrain, :-1, 0].unsqueeze(-1).unsqueeze(-1) 
    u0_train = u0_train.repeat([1, 1, T, 1])

    u_test = u[-ntest:, :-1, 0:T:sub_t]
    u0_test = u[-ntest:, :-1, 0].unsqueeze(-1).unsqueeze(-1)
    u0_test = u0_test.repeat([1, 1, T, 1])

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, u_test), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


#===========================================================================
# Training and Testing functionalities
#===========================================================================
def eval_fno_1d(model, test_dl, myloss, batch_size, device):

    ntest = len(test_dl.dataset)
    test_loss = 0.
    with torch.no_grad():
        for u0_, xi_, u_ in test_dl:    
            loss = 0.       
            xi_, u_ = xi_.to(device), u_.to(device)
            u_pred = model(xi_)
            u_pred = u_pred[..., 0]
            loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1)) 
            test_loss += loss.item()
    print('Test Loss: {:.6f}'.format(test_loss / ntest))
    return test_loss / ntest

def train_fno_1d(model, train_loader, test_loader, device, myloss, batch_size=20, epochs=5000, learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, plateau_patience=None, plateau_terminate=None, print_every=20, checkpoint_file='checkpoint.pt'):


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if plateau_patience is None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=plateau_patience, threshold=1e-6, min_lr=1e-7)
    if plateau_terminate is not None:
        early_stopping = EarlyStopping(patience=plateau_terminate, verbose=False, path=checkpoint_file)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    losses_train = []
    losses_test = []

    try:
            
        for ep in range(epochs):

            model.train()
            
            train_loss = 0.
            for xi_, u_ in train_loader: 

                loss = 0.
                xi_ = xi_.to(device)
                u_ = u_.to(device)

                u_pred = model(xi_)
                u_pred = u_pred[..., 0]
                loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            test_loss = 0.
            with torch.no_grad():
                for xi_, u_ in test_loader:
                    
                    loss = 0.
                    
                    xi_ = xi_.to(device)
                    u_ = u_.to(device)

                    u_pred = model(xi_)
                    u_pred = u_pred[..., 0]
                    loss = myloss(u_pred[..., 1:].reshape(batch_size, -1), u_[..., 1:].reshape(batch_size, -1))

                    test_loss += loss.item()

            if plateau_patience is None:
                scheduler.step()
            else:
                scheduler.step(test_loss/ntest)
            if plateau_terminate is not None:
                early_stopping(test_loss/ntest, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if ep % print_every == 0:
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Test Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        return model, losses_train, losses_test
    
    except KeyboardInterrupt:

        return model, losses_train, losses_test


def hyperparameter_search(train_dl, val_dl, test_dl, T, d_h=[32], iter=[1,2,3], modes1=[32,64], modes2=[32,64], epochs=500, print_every=20, lr=0.025, plateau_patience=100, plateau_terminate=100, log_file ='log_nspde', checkpoint_file='checkpoint.pt', final_checkpoint_file='final.pt'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparams = list(itertools.product(d_h, iter, modes1, modes2))

    loss = LpLoss(size_average=False)
    
    fieldnames = ['d_h', 'L', 'modes1', 'modes2', 'nb_params', 'loss_train', 'loss_val', 'loss_test']
    with open(log_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        

    best_loss_val = 1000.

    for (_dh, _iter, _modes1, _modes2) in hyperparams:
        
        print('\n dh:{}, iter:{}, modes1:{}, modes2:{}'.format(_dh, _iter, _modes1, _modes2))

        model = FNO_space1D_time(modes1=_modes1, modes2=_modes2, width=_dh, T=T, L=_iter).cuda()

        nb_params = count_params(model)
        
        print('\n The model has {} parameters'. format(nb_params))

        # Train the model. The best model is checkpointed.
        
        _, _, _ = train_fno_1d(model, train_dl, val_dl, device, loss, batch_size=20, epochs=epochs, learning_rate=lr, scheduler_step=500, scheduler_gamma=0.5, plateau_patience=plateau_patience, plateau_terminate=plateau_terminate, print_every=print_every, checkpoint_file=checkpoint_file)
        # load the best trained model 
        model.load_state_dict(torch.load(checkpoint_file))
        
        # compute the test loss 
        loss_test = eval_fno_1d(model, test_dl, loss, 20, device)

        # we also recompute the validation and train loss
        loss_train = eval_fno_1d(model, train_dl, loss, 20, device)
        loss_val = eval_fno_1d(model, val_dl, loss, 20, device)

        # if this configuration of hyperparameters is the best so far (determined wihtout using the test set), save it 
        if loss_val < best_loss_val:
            torch.save(model.state_dict(), final_checkpoint_file)
            best_loss_val = loss_val

        # write results
        with open(log_file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([_dh, _iter, _modes1, _modes2, nb_params, loss_train, loss_val, loss_test])

