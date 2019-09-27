import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import Config as cfg
from math import pi


class RotateTrans(nn.Module):
    def __init__(self):
        super(RotateTrans, self).__init__()

        self.rad = torch.nn.Parameter(2*torch.rand(1)-1)

    def forward(self, x):
        rad = torch.clamp(self.rad, -1, 1)
        cos = torch.cos(pi*rad)
        sin = torch.sin(pi*rad)

        theta_tag = torch.zeros([2, 3]).cuda()
        theta_tag[0, 0] = cos
        theta_tag[1, 1] = cos
        theta_tag[1, 0] = sin
        theta_tag[0, 1] = -1 * sin

        theta_tag = theta_tag.view(-1, 2, 3)
        theta_tag = theta_tag.expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta_tag, x.size())
        x = F.grid_sample(x, grid)

        return x


class ConstTrans(nn.Module):
    def __init__(self, mask):
        super(ConstTrans, self).__init__()

        q = torch.randn((1, 2, 3))
        self.mask = mask.cuda()
        theta_const = torch.nn.Parameter(q)
        theta_const.requires_grad = True
        self.theta_const = theta_const.expand(cfg.BATCH_SIZE, -1, -1).cuda()

    def forward(self, x):
        qqq = torch.clamp(self.theta_const, -1, 1)
        qqq = qqq * self.mask
        grid = F.affine_grid(qqq[0:x.shape[0], :, :], x.size())
        x = F.grid_sample(x, grid)

        return x


class EqScaleTrans(nn.Module):
    def __init__(self):
        super(EqScaleTrans, self).__init__()

        self.scale_const = torch.nn.Parameter(torch.rand(1))

    def forward(self, x):
        qqq = torch.clamp(self.scale_const, -1, 1)
        qqq = torch.clamp(1/qqq, -2, 2)
        qqq = 1/qqq

        theta_tag = torch.zeros([2, 3]).cuda()
        theta_tag[0, 0] = qqq
        theta_tag[1, 1] = qqq

        theta_tag = theta_tag.view(-1, 2, 3)
        theta_tag = theta_tag.expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta_tag, x.size())
        x = F.grid_sample(x, grid)

        return x
