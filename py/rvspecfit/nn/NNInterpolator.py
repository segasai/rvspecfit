import torch.nn as tonn
import torch.nn.modules.activation as tonnact
import torch
from collections import OrderedDict
import numpy as np
import warnings

# Identifies our custom torch checkpoint container for NNInterpolator.
CHECKPOINT_MAGIC = 'rvspecfit.nn_interpolator'
# Version of checkpoint payload structure (top-level keys/format).
CHECKPOINT_VERSION = 1


class NNInterpolator(tonn.Module):
    # Version of NNInterpolator architecture semantics.
    # Bump when code changes make old weights unsafe/incompatible.
    ARCHITECTURE_VERSION = 1

    def __init__(self,
                 indim=None,
                 nlayers=None,
                 width=None,
                 npc=None,
                 npix=None,
                 withbn=True,
                 nonlinearity='SiLU'):
        super(NNInterpolator, self).__init__()
        self.indim = indim
        self.nlayers = nlayers
        self.width = width
        self.npc = npc
        self.npix = npix
        self.withbn = withbn
        self.nonlinearity = nonlinearity
        self.initLayers()

    def initLayers(self):
        # Layer declaration
        shapes = [
            (self.indim, self.width)
        ] + [(self.width, self.width)] * self.nlayers + [(self.width, self.npc)
                                                         ]
        # self.L0 = tonn.Linear(self.indim, self.width)
        layer_dict = OrderedDict()
        NL = getattr(tonnact, self.nonlinearity)
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
            withbn = self.withbn
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


def build_checkpoint(model):
    # Wrapper format: metadata + raw torch state_dict.
    return dict(checkpoint_magic=CHECKPOINT_MAGIC,
                checkpoint_version=CHECKPOINT_VERSION,
                nn_arch_version=model.ARCHITECTURE_VERSION,
                state_dict=model.state_dict())


def _extract_state_dict(model, checkpoint, allow_legacy=False, source='checkpoint'):
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # 1) Confirm this payload is our checkpoint type.
        if checkpoint.get('checkpoint_magic') != CHECKPOINT_MAGIC:
            raise RuntimeError(f'Invalid checkpoint magic in {source}')
        # 2) Confirm loader understands this checkpoint schema.
        if checkpoint.get('checkpoint_version') != CHECKPOINT_VERSION:
            raise RuntimeError(f'Unsupported checkpoint version in {source}')
        # 3) Confirm weights were produced for compatible NN code.
        nn_arch_version = checkpoint.get('nn_arch_version')
        if nn_arch_version != model.ARCHITECTURE_VERSION:
            raise RuntimeError(
                f'NNInterpolator architecture version mismatch in {source}: '
                f'weights={nn_arch_version} code={model.ARCHITECTURE_VERSION}')
        return checkpoint['state_dict']

    # Legacy format: raw state_dict saved via torch.save(model.state_dict()).
    if isinstance(checkpoint, dict):
        if not allow_legacy:
            raise RuntimeError(
                f'Legacy NNInterpolator checkpoint format in {source} has no '
                'version metadata; refusing to load')
        warnings.warn(
            f'Loading legacy NNInterpolator checkpoint without version '
            f'metadata from {source}. Please re-save/retrain to get strict '
            'compatibility checks.',
            RuntimeWarning)
        return checkpoint

    raise RuntimeError(f'Unsupported checkpoint payload in {source}')


def save_checkpoint(model, path):
    torch.save(build_checkpoint(model), path)


def load_checkpoint(model,
                    path,
                    map_location=None,
                    allow_legacy=False,
                    weights_only=True):
    # `weights_only=True` (newer torch) avoids loading arbitrary pickled code.
    # Keep a fallback for older torch versions that do not support this kwarg.
    load_kwargs = {}
    if map_location is not None:
        load_kwargs['map_location'] = map_location
    try:
        checkpoint = torch.load(path, weights_only=weights_only, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(path, **load_kwargs)
    state_dict = _extract_state_dict(model,
                                     checkpoint,
                                     allow_legacy=allow_legacy,
                                     source=path)
    model.load_state_dict(state_dict)


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
