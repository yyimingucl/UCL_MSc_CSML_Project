import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
All the three kernels are from 
"Measuring Sample Quality with Kernels, Jackson Gorham&Lester Mackey (2020)"
All the three kernels are modified based on selecting median of the squared Euclidean distance 
between pairs of sample points as the bandwidth h

More details can be found at 
Measuring Sample Quality with Kernels : https://arxiv.org/pdf/1703.01717.pdf
SVGD: https://arxiv.org/pdf/1608.04471.pdf
"""

def _square_dist(x):
    # * x: [num_sample, feature_dim]
    xxT = torch.mm(x, x.t())
    # * xxT: [num_sample, num_sample]
    xTx = xxT.diag()
    # * xTx: [num_sample]
    return xTx + xTx.unsqueeze(1) - 2. * xxT


def Gaussian_Kxx_dxKxx(x, num_agent):
    #Adpative Gaussian Kernel
    #k(x,y) = exp(-||x-y||_2^2/h)

    square_dist = _square_dist(x)
    # * h = bandwidth = 2 * (med ^ 2)
    bandwidth = 2 * square_dist.median() / math.log(num_agent)
    Kxx = torch.exp(-1. / bandwidth * square_dist)

    dxKxx = 2 * (Kxx.sum(1).diag() - Kxx).matmul(x) / bandwidth

    return Kxx, dxKxx


def IMQ_Kxx_dxKxx(x, num_agent):
    #Invert MultiQuadratic Kernel
    #k(x,y) = (1+1/h*||x-y||_2^2)^(-1/2)

    square_dist = _square_dist(x)
    # * h = bandwidth = 2 * (med ^ 2)
    bandwidth = 2 * square_dist.median() / math.log(num_agent)
    Kxx = 1./torch.sqrt(1+square_dist/bandwidth)
    Kxx_3 = torch.matrix_power(Kxx,3)

    dxKxx = (Kxx_3.sum(1).diag() - Kxx_3).matmul(x) / bandwidth
    return Kxx, dxKxx


def Matern_Kxx_dxKxx(x, num_agent):
    #Matern Radial Kernel 
    #k(x,y) = (1+sqrt(h)*||x-y||_2) * exp(-sqrt(h)||x-y||_2)

    abs_dist = torch.sqrt(_square_dist(x))
    # * h = bandwidth = 2 * (med ^ 2)

    h = torch.sqrt(2 * abs_dist.median() / math.log(num_agent))

    exp = torch.exp(-abs_dist*h)
    Kxx = (1+abs_dist*h) * exp
    dxKxx = h**2 * (1+h) * (exp.sum(1).diag() - exp).matmul(x)

    return Kxx, dxKxx

