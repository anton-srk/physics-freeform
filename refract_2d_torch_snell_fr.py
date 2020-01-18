#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:00:27 2019

@author: anton
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import pickle
from pykeops.torch import LazyTensor

# amplitude of the surface
EPSILON = 0.001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
dtype = torch.cuda.FloatTensor
# dtype = torch.float32
torch.backends.cudnn.benchmark = True

# overall number of the screen points
NUMPTS = 1000 * 1000

use_cuda = torch.cuda.is_available()
# dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


def laplacian_kernel(x, y, sigma=.1):
    '''Laplacian kernel function'''
    # (M, 1, 1)
    x_i = LazyTensor(x[:, None, :])
    # (1, N, 1)
    y_j = LazyTensor(y[None, :, :])
    # (M, N) symbolic matrix of squared distances
    D_ij = ((x_i - y_j)**2).sum(-1)
    # (M, N) symbolic Laplacian kernel matrix
    return (-D_ij.sqrt()/sigma).exp()


def plview(tens, pts=100):
    '''Plot a pytorch tensor data'''
    plt.figure()
    plt.imshow(tens.view(pts, pts).detach().cpu().numpy())
    plt.colorbar()
    plt.show()


def plscatter(tens, alpha=0.01):
    '''Scatter plot a pytorch tensor'''
    return plt.scatter(tens[:, 0].detach().cpu().numpy(),
                       tens[:, 1].detach().cpu().numpy(), alpha=alpha)


def limit(z):
    '''Helper function to limit the size of the screen'''
    if ((z < 0) or (z > (int(np.sqrt(NUMPTS))-1))):
        return 0
    else:
        return z


def n(r, a, x):
    '''Refractive index'''
    r2d = r[:, :2].contiguous().to(device)
    K_tx = laplacian_kernel(r2d, x)
    mean_t = K_tx@a
    sz = torch.tensor(np.sqrt(NUMPTS)).type(torch.long)
    mean_t = mean_t.view(sz, sz)

    cr1, cr2 = (int(np.sqrt(NUMPTS)/4)*r[:, 0]).type(torch.long),
    (int(np.sqrt(NUMPTS)/4)*r[:, 1]).type(torch.long)

    cr1, cr2 = cr1.cpu().apply_(limit), cr2.cpu().apply_(limit)

    if use_cuda:
        cr1, cr2 = cr1.cuda(), cr2.cuda()

    screen = mean_t[cr1, cr2].type(dtype)

    return (-EPSILON*screen)


def diff_y(y, a, x):
    ''' Calculates the gradient based on the stacked coordinate and
    speed vectors tensor'''
    t_diff = y[:, :2].clone().detach().requires_grad_(True)
    n_t = n(t_diff, a, x)
    n_t.backward(torch.ones_like(n_t), retain_graph=True)
    grd = t_diff.grad

    return grd


def calc_pt(a, x):
    ''' Calculates vectors and gradients'''
    Xr = Yr = torch.linspace(0, 4,
                             int(np.sqrt(NUMPTS)),
                             requires_grad=True).type(dtype)
    Xr, Yr = torch.meshgrid(Xr, Yr)
    xr = torch.stack((Xr.contiguous().view(-1), Yr.contiguous().view(-1)),
                     dim=1)
    # coordinate vectors
    r_0 = torch.cat([xr,
                     torch.zeros(NUMPTS, 1, requires_grad=True).
                     type(dtype)], dim=1)
    # speed vectors
    v_0 = torch.cat([torch.zeros(NUMPTS, 2, requires_grad=True),
                     torch.ones(NUMPTS, 1, requires_grad=True)],
                    dim=1).type(dtype)

    # concatenating speed and coordinate vectors
    y = torch.cat((r_0, v_0), dim=1)

    # calculating the gradient
    gr2d = diff_y(y, a, x)[:, :2]

    # calculating the vector normal to the surface
    norm_t = torch.cat([gr2d,
                        -1*torch.ones(gr2d.shape[0]).
                        unsqueeze(-1).to(device)], dim=1)
    norm_t = (norm_t /
              torch.sqrt(norm_t[:, 0]**2 +
                         norm_t[:, 1]**2 +
                         norm_t[:, 2]**2).unsqueeze(-1))

    n_out = 1.5

    # coefficient defining the change in direction of the vectors passing
    # through the target
    Kfun = ((n_out-torch.sqrt(1+gr2d[:, 0]**2+gr2d[:, 1]**2)*(1-n_out**2)) /
            (1+gr2d[:, 0]**2+gr2d[:, 1]**2))

    return r_0[:, 0], r_0[:, 1], gr2d[:, 0], gr2d[:, 1], Kfun


def main():

    # loading a "noisy" surface profile
    with open('/home/anton/Documents/surface_prof_noisy_0.25_50', 'rb') as opened_file:
        a = pickle.load(opened_file).to(device)

    X0 = Y0 = torch.linspace(0, 4, 100, requires_grad=True).type(dtype)
    X0, Y0 = torch.meshgrid(X0, Y0)
    x = torch.stack((X0.contiguous().view(-1),
                     Y0.contiguous().view(-1)), dim=1)

    start_time = time.time()
    r_0x, r_0y, gr2dx, gr2dy, Kfun = calc_pt(a, x)

    distances = [1, 2, 5, 10, 15, 20, 50]
    scans = {}

    for i in distances:

        r_1x = r_0x + i * (gr2dx) * Kfun / (1-Kfun)
        r_1y = r_0y + i * (gr2dy) * Kfun / (1-Kfun)

        x = torch.stack([r_1x, r_1y]).t().contiguous()

        # Calculating the laplacian kernel for x
        x_i = LazyTensor(x[:, None, :])
        y_j = LazyTensor(x[None, :, :])
        D_ij = ((x_i - y_j)**2).sum(dim=2)
        K_ij = (-D_ij.sqrt()/0.075).exp()

        b = K_ij.sum(dim=1)

        plview(b, 1000)
        plt.savefig('noisy_dist_'+str(i)+'.png')

        scans[i] = b.view(1000, 1000)[:, 500].detach().cpu().numpy()

    with open('noisy_scans', 'wb') as handle:
        pickle.dump(scans, handle)

    print("--- %s seconds ---" % (time.time()-start_time))


main()
