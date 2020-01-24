import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
N = 100 if use_cuda else 1000  # Number of samples


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


def main():
    # loading the surface profile
    with open('data/z_profile', 'rb') as opened_file:
        z2t = pickle.load(opened_file).to(device)
    # loading the noise screen
    with open('data/noise_screen_50', 'rb') as opened_file:
        noise = pickle.load(opened_file).to(device)

    # processing the surface profile and adding the noise
    z2t = torch.nn.functional.interpolate(
            z2t.view(1001, 1001).unsqueeze(0).unsqueeze(0), size=100)
    z2t = z2t + 0.25*noise

    # creating a meshgrid
    X = Y = torch.linspace(0, 1, 100, requires_grad=True).type(dtype)
    X, Y = torch.meshgrid(X, Y)
    x = torch.stack((X.contiguous().view(-1), Y.contiguous().view(-1)), dim=1)

    b = z2t.squeeze().view(10000, 1)

    # Ridge regularization
    alpha = 0

    start = time.time()

    K_xx = laplacian_kernel(x, x)
    a = K_xx.solve(b, alpha=alpha)

    end = time.time()
    # dump the resulting interpolation profile to a file
    with open('data/surface_prof_noisy', 'wb') as opened_file:
        pickle.dump(a, opened_file)

    print('Time to perform an RBF \
          interpolation with {:,} samples \
          in 2D: {:.5f}s'.format(N, end - start))
    # plotting the resulting noisy surface
    X = Y = torch.linspace(0, 1, 100, requires_grad=True).type(dtype)
    X, Y = torch.meshgrid(X, Y)
    t = torch.stack((X.contiguous().view(-1), Y.contiguous().view(-1)), dim=1)

    K_tx = laplacian_kernel(t, x)
    mean_t = K_tx@a
    mean_t = mean_t.view(100, 100)

    plt.figure(figsize=(8, 8))
    plt.imshow(mean_t.detach().cpu().numpy()[::-1, :],
               interpolation="bilinear", cmap="coolwarm")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


main()
