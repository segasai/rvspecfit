import torch.nn as tonn
import torch
from collections import OrderedDict
import numpy as np


class NNInterpolator(tonn.Module):

    def __init__(self,
                 indim=None,
                 nlayers=None,
                 width=None,
                 npc=None,
                 npix=None):
        super(NNInterpolator, self).__init__()
        self.indim = indim
        self.nlayers = nlayers
        self.width = width
        self.npc = npc
        self.npix = npix
        self.initLayers()

    def initLayers(self):
        # Layer declaration
        shapes = [
            (self.indim, self.width)
        ] + [(self.width, self.width)] * self.nlayers + [(self.width, self.npc)
                                                         ]
        # self.L0 = tonn.Linear(self.indim, self.width)
        layer_dict = OrderedDict()
        # NL = tonn.Tanh
        # NL = tonn.LeakyReLU
        NL = tonn.ReLU
        NL = tonn.SiLU
        # NL = tonn.CELU
        # NL = tonn.LeakyReLU
        """ sequence here is
        * is (indim x width) layer with bias
        * Nonlinearity

        * (width x width) layer with no bias
        * batchnorm
        * nonlinearity
        * The last 3 lements repeated nlayers times

        * (width x npc) layer with bias
        * nonlinearity
        * (npc x npix) layers
        """
        batchnorm_after_nl = True

        for i in range(len(shapes)):
            withbn = True
            if i == 0 or i == len(shapes) - 1:
                withbn = False
            if batchnorm_after_nl:
                bias = True
            else:
                # if batchnorm is just after linear
                # bias is not needed
                bias = not withbn
            curl = tonn.Linear(shapes[i][0], shapes[i][1], bias=bias)

            layer_dict['lin_%d' % i] = curl
            if withbn:
                curbn = tonn.BatchNorm1d(shapes[i][1])
                if batchnorm_after_nl:
                    layer_dict['nl_%d' % i] = NL()
                    layer_dict['bn_%d' % i] = curbn
                else:
                    layer_dict['bn_%d' % i] = curbn
                    layer_dict['nl_%d' % i] = NL()
            else:
                layer_dict['nl_%d' % i] = NL()

        self.model = tonn.Sequential(layer_dict)
        self.pc_layer = tonn.Linear(self.npc, self.npix)

    def forward(self, curx):
        curx = curx.view(-1, self.indim)
        curx = self.pc_layer(self.model(curx))
        return curx


class Mapper:

    def __init__(self, M, S, log_ids=[0]):
        self.M = M
        self.S = S
        self.log_ids = log_ids

    def forward(self, x):
        x1 = np.asarray(x, dtype=np.float32)
        y = x1 * 1
        for ii in self.log_ids:
            y[..., ii] = np.log10(x1[..., ii])
        return (y - self.M) / self.S
