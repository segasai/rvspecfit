from . import NNInterpolator
import torch
import scipy.spatial
import numpy as np
import os
import logging

device_env_name = 'RVS_NN_DEVICE'
if device_env_name in os.environ:
    device_env = os.environ[device_env_name]
else:
    device_env = None


class RVSInterpolator:

    def __init__(self, kwargs):
        self.nni = NNInterpolator.NNInterpolator(**kwargs['class_kwargs'])

        if device_env is None:
            self.device = torch.device(kwargs['device'])
        else:
            self.device = torch.device(device_env)
        self.nni.load_state_dict(
            torch.load(kwargs['template_lib'] + '/' + kwargs['nn_file'],
                       map_location=self.device,
                       weights_only=True))
        # self.device = list(self.nni.children())[0][0].pc_layer.weight.device

        self.nni.to(self.device)
        logging.debug(f'RVS NN interpolator device: {self.device}')
        self.nni.eval()

    def __call__(self, x):
        with torch.inference_mode():
            ret = self.nni(
                torch.tensor(x, dtype=torch.float32).to(
                    self.device)).cpu().detach().numpy().astype(np.float64)
            # prevent formal overflow
            return np.exp(np.clip(ret, -300, 300)).flatten()


class OutsideInterpolator:

    def __init__(self, kwargs0):
        kwargs = kwargs0['outside_kwargs']
        pts = kwargs['pts']
        # separate first two dims
        # and last two dims
        xdim2 = pts[:, :2]
        ydim2 = pts[:, 2:]
        xconv = scipy.spatial.ConvexHull(xdim2)
        xvec = xdim2[xconv.vertices]
        yconv = scipy.spatial.ConvexHull(ydim2)
        yvec = ydim2[yconv.vertices]
        self.xtriang = scipy.spatial.Delaunay(xvec)
        self.ytriang = scipy.spatial.Delaunay(yvec)
        self.xeqs = xconv.equations
        self.yeqs = yconv.equations

    def __call__(self, p):
        ret = 0
        if self.xtriang.find_simplex(p[:2]) < 0 or self.ytriang.find_simplex(
                p[2:]) < 0:
            d_x = np.max(self.xeqs[:, :-1] @ p[:2] + self.xeqs[:, -1])
            d_y = np.max(self.yeqs[:, :-1] @ p[2:] + self.yeqs[:, -1])
            # ret = self.tree.query(p, 4)[0].mean()
            return max(max(d_x, d_y), 0)
        return ret

    @staticmethod
    def generate_pts(vecs):
        return vecs
        # conv = scipy.spatial.ConvexHull(vecs)
        # return vecs[conv.vertices]
