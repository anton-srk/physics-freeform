#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:15:46 2019

@author: anton
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle


def plview(tens, pts=100):
    '''Function used to plot tensors'''
    plt.figure()
    plt.imshow(tens.view(pts, pts).detach().cpu().numpy())
    plt.colorbar()
    plt.show()


def noise_gen(sigma):
    ''' Noise generating function.
    Creates a gaussian noise screen with predefined parameter
    sigma '''
    r_z = torch.zeros(512, 512)
    r_f = torch.rfft(r_z, 2, normalized=True, onesided=False)

    for i in range(512):
        for j in range(512):
            r_f[i, j, :] = np.random.rand()*np.exp(-((i-256)**2 +
                                                   (j-256)**2)/sigma)

    r_rf = torch.irfft(r_f, 2, normalized=True, onesided=False)
    r_cut = r_rf[50:450, 50:450]
    r_cut = r_cut.unsqueeze(0).unsqueeze(0)
    r_interp = torch.nn.functional.interpolate(r_cut, size=100).squeeze()
    r_interp = r_interp - r_interp.min()
    r_interp = r_interp / r_interp.max()

    return(r_interp)


def main():

    sigmas = [50, 500, 5000]

    for s in sigmas:
        n = noise_gen(s)
        with open('data/noise_screen_'+str(s), 'wb') as handle:
            pickle.dump(n, handle)


main()
