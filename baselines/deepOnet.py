import torch
import numpy as np
import torch.nn as nn
from .utils import UnitGaussianNormalizer

class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class DeepONetCP(nn.Module):
    def __init__(self, branch_layer, trunk_layer):
        super(DeepONetCP, self).__init__()
        self.branch = DenseNet(branch_layer, nn.ReLU, nn.Tanh)
        self.trunk = DenseNet(trunk_layer, nn.ReLU, nn.Tanh)

    def forward(self, u0, grid):
        a = self.branch(u0)
        # batchsize x width
        b = self.trunk(grid)
        # N x width
        return torch.einsum('bi,ni->bn', a, b)


#===========================================================================
# Data Loaders for Neural SPDE
#===========================================================================

def dataloader_deeponet_1d_xi(u, xi, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, normalizer=False, dataset=None):

    if dataset=='phi41':
        T, sub_t, dim_t = 51, 1, 50
    elif dataset=='wave':
        T, sub_t = (u.shape[-1]+1)//2, 5  #TODO: dim_t

    u_train = u[:ntrain, :dim_x, 1:T:sub_t].reshape(ntrain, -1)
    xi_train = torch.diff(xi[:ntrain, :-1, 0:T:sub_t],dim=-1)
    xi_train = torch.cat([torch.zeros_like(xi_train[..., 0].unsqueeze(-1)), xi_train], dim=-1).reshape(ntrain, -1)

    u_test = u[-ntest:, :dim_x, 1:T:sub_t].reshape(ntest, -1)
    xi_test = torch.diff(xi[-ntest:, :-1, 0:T:sub_t],dim=-1)
    xi_test = torch.cat([torch.zeros_like(xi_test[..., 0].unsqueeze(-1)), xi_test], dim=-1).reshape(ntest, -1)

    if normalizer:
        xi_normalizer = UnitGaussianNormalizer(xi_train)
        xi_train = xi_normalizer.encode(xi_train)
        xi_test = xi_normalizer.encode(xi_test)

        u_normalizer = UnitGaussianNormalizer(u_train)
        u_train = u_normalizer.encode(u_train)
        u_test = u_normalizer.encode(u_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xi_test, u_test), batch_size=batch_size, shuffle=False)

    gridx = torch.tensor(np.linspace(0, 1, dim_x), dtype=torch.float)
    gridx = gridx.reshape(1, dim_x, 1, 1).repeat([batch_size, 1, dim_t, 1])
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float)
    gridt = gridt.reshape(1, 1, dim_t, 1).repeat([batch_size, dim_x, 1, 1])
    grid =  torch.cat((gridx, gridt), dim=-1)[0].reshape(dim_x * dim_t, 2)

    return train_loader, test_loader, u_normalizer, grid

def dataloader_deeponet_1d_u0(u, ntrain=1000, ntest=200, T=51, sub_t=1, batch_size=20, dim_x=128, normalizer=False, dataset=None):

    if dataset=='phi41':
        T, sub_t, dim_t = 51, 1, 50
    elif dataset=='wave':
        T, sub_t = (u.shape[-1]+1)//2, 5

    u0_train = u[:ntrain, :-1, 0].reshape(ntrain, -1)
    u_train = u[:ntrain, :-1, 1:T:sub_t].reshape(ntrain, -1)

    u0_test = u[-ntest:, :-1, 0].reshape(ntest, -1)
    u_test = u[-ntest:, :-1, 1:T:sub_t].reshape(ntest, -1)

    if normalizer:
        u_normalizer = UnitGaussianNormalizer(u_train)
        u_train = u_normalizer.encode(u_train)
        u_test = u_normalizer.encode(u_test)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_train, u_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u0_test, u_test), batch_size=batch_size, shuffle=False)

    gridx = torch.tensor(np.linspace(0, 1, dim_x), dtype=torch.float)
    gridx = gridx.reshape(1, dim_x, 1, 1).repeat([batch_size, 1, dim_t, 1])
    gridt = torch.tensor(np.linspace(0, 1, dim_t), dtype=torch.float)
    gridt = gridt.reshape(1, 1, dim_t, 1).repeat([batch_size, dim_x, 1, 1])
    grid =  torch.cat((gridx, gridt), dim=-1)[0].reshape(dim_x * dim_t, 2)

    return train_loader, test_loader, u_normalizer, grid

#===========================================================================
# Training functionalities
#===========================================================================

def train_deepOnet_1d(model, train_loader, test_loader, grid, u_normalizer, device, myloss, batch_size=20, epochs=5000, learning_rate=0.001, scheduler_step=100, scheduler_gamma=0.5, print_every=20):
    
    grid = grid.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)
    
    losses_train = []
    losses_test = []

    try:
        for ep in range(epochs):

            model.train()
            
            train_loss = 0.
            for u0_ , u_ in train_loader:
                
                loss = 0.

                u0_ = u0_.to(device)
                u_ = u_.to(device)
                
                u_pred = model(u0_, grid)

                if u_normalizer is not None:
                    u_pred = u_normalizer.decode(u_pred.cpu())
                    u_ = u_normalizer.decode(u_.cpu())
                
                loss = myloss(u_pred.reshape(batch_size, -1), u_.reshape(batch_size, -1))

                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            test_loss = 0.
            with torch.no_grad():
                for u0_, u_ in test_loader:
                    
                    loss = 0.
                    
                    u0_ = u0_.to(device)
                    u_ = u_.to(device)

                    u_pred = model(u0_, grid)
                    
                    if u_normalizer is not None:
                        u_pred = u_normalizer.decode(u_pred.cpu())
                        u_ = u_normalizer.decode(u_.cpu())

                    loss = myloss(u_pred.reshape(batch_size, -1), u_.reshape(batch_size, -1))

                    test_loss += loss.item()

            scheduler.step()

            if ep % print_every == 0:
        
                losses_train.append(train_loss/ntrain)
                losses_test.append(test_loss/ntest)
                print('Epoch {:04d} | Total Train Loss {:.6f} | Total Test Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

        return model, losses_train, losses_test
    
    except KeyboardInterrupt:

        return model, losses_train, losses_test