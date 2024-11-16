import os
import sys
import argparse
import time
import rvspecfit
import torch.utils.data as toda
import torch.nn.functional as tofu
import sklearn.decomposition as skde
import numpy as np
import torch
from .NNInterpolator import Mapper, NNInterpolator
from rvspecfit import serializer

git_rev = rvspecfit.__version__


def getData(dir, setup, log_ids=[0]):

    fname = f'{dir}/specs_{setup}.h5'
    dat = serializer.load_dict_from_hdf5(fname)
    lam = dat['lam']
    vecs = dat['vec'].T[:]
    dats = dat['specs']

    xvecs = vecs * 1
    for ii in log_ids:
        xvecs[:, ii] = np.log10(vecs[:, ii])
    M = xvecs.mean(axis=0)
    S = xvecs.std(axis=0)
    mapper = Mapper(M, S, log_ids)

    # vecs[:, 0] = np.log10(vecs[:, 0])

    vecs_trans = mapper.forward(vecs)
    return lam, vecs_trans, dats, mapper, vecs


def getSchedOptim(optim):
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                       factor=0.5,
                                                       patience=20,
                                                       eps=1e-9,
                                                       threshold=1e-5)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim,50)
    return sched


def get_predictions(myint, Tvecs0, dev, batch):
    alldat = toda.TensorDataset(Tvecs0)
    pred = []
    batchdatF = toda.DataLoader(
        alldat,
        batch_size=batch,
        shuffle=False,
    )
    myint.train(mode=False)
    with torch.inference_mode():
        for Tvecs in batchdatF:
            Tvecs = Tvecs[0].to(dev)
            Rfinal = myint(Tvecs).detach().cpu().numpy()
            pred.append(Rfinal)
    pred = np.concatenate(pred)
    return pred


def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--random_pca', action='store_true', default=False)
    parser.add_argument('--pca_init',
                        action='store_true',
                        default=False,
                        help='initialize with pca')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--dir', type=str, default='./')
    parser.add_argument('--nlayers',
                        type=int,
                        default=2,
                        help='number of inner fc layers')
    parser.add_argument('--revision', default='', help='Revision string')
    parser.add_argument('--width', type=int, default=256, help='Network width')
    parser.add_argument('--npc', type=int, default=200)
    parser.add_argument('--learning_rate0', type=float, default=1e-3)
    parser.add_argument('--parnames', type=str, default='teff,logg,feh,alpha')
    parser.add_argument('--log_ids', type=str, default='0')
    parser.add_argument('--mask_ids', type=str, default=None)
    parser.add_argument('--setup', type=str)
    parser.add_argument('--batch', type=int, default=100)
    args = parser.parse_args(args)
    log_ids = [int(_) for _ in (args.log_ids).split(',')]
    setup = args.setup
    revision = args.revision
    parnames = args.parnames.split(',')
    directory = args.dir
    pca_init = args.pca_init
    random_pca = args.random_pca
    if args.cpu:
        train_dev = torch.device('cpu')
    else:
        train_dev = torch.device('cuda')
    device_name = 'cuda'  # for inference

    print('using', train_dev)
    np.random.seed(22)
    torch.manual_seed(343432323)
    rstate = np.random.default_rng(44)

    # vecs are transformed already
    lam, vecs, dats, mapper, vecs_orig = getData(directory,
                                                 setup,
                                                 log_ids=log_ids)
    D_0 = np.mean(dats, axis=0)
    SD_0 = np.std(dats, axis=0)
    tD_0 = torch.tensor(D_0)
    tSD_0 = torch.tensor(SD_0)
    nspec, npix = dats.shape
    # ids = np.arange(nspec)
    print(npix, nspec)

    if args.mask_ids is not None:
        mask_ids = [int(_) for _ in args.mask_ids.split(',')]
    else:
        mask_ids = None

    width = args.width
    indim = vecs.shape[1]
    nlayers = args.nlayers
    npc = args.npc
    lr0 = args.learning_rate0
    # nstack = 1
    batch = args.batch
    validation_frac = 0.05
    validation = args.validation
    if validation:
        train_set = rstate.uniform(size=len(dats)) > validation_frac
        validation_set = ~train_set
    else:
        train_set = np.ones(len(dats), dtype=bool)
    if mask_ids is not None:
        print('masking', mask_ids)
        mask = np.zeros(nspec, dtype=bool)
        mask[mask_ids] = True
        train_set = train_set & (~mask)
    print(train_set.sum())
    kwargs = dict(indim=indim,
                  npc=npc,
                  nlayers=nlayers,
                  width=width,
                  npix=npix,
                  withbn=False)
    myint = NNInterpolator(**kwargs)

    statefile_path = f'{directory}/tmp_state_{setup}.sav'
    finalfile = f'NNstate_{setup}.sav'
    finalfile_path = f'{directory}/{finalfile}'

    restore = args.resume
    if restore:
        if os.path.exists(statefile_path):
            try:
                myint.load_state_dict(torch.load(statefile_path))
                print('restoring', statefile_path)
            except RuntimeError:
                print('failed to restore')
                restore = False
        else:
            restore = False

    # we just normalise loss by that to have it around 1
    # if the network is not doing anything
    spread0 = np.std(dats - np.mean(dats, axis=0))

    if not restore:
        if pca_init:
            pca = skde.PCA(n_components=npc)
            pca.fit(dats[train_set])
            if random_pca:
                pca_comps = rstate.normal(size=(npc, npc)) @ pca.components_
            else:
                pca_comps = pca.components_
            pca_comps = pca_comps / np.sqrt(
                (pca_comps**2).sum(axis=1))[:, None]
            myint.pc_layer.weight.data[:, :] = torch.tensor(
                pca_comps.T) / tSD_0.view(npix, 1)
            myint.pc_layer.bias.data[:] = torch.tensor(pca.mean_.T) * 0
            xpred = pca.inverse_transform(pca.transform(dats[train_set]))
            loss0 = np.abs(dats[train_set] - xpred).mean()
            print('loss0', loss0 / spread0)

    myint.to(train_dev)
    tD_0 = tD_0.to(train_dev)
    tSD_0 = tSD_0.to(train_dev)

    Tvecs0 = torch.FloatTensor(data=vecs)
    Tdat0 = torch.as_tensor(dats)
    batch_on_dev = False

    if not batch_on_dev:
        Tvecs0 = Tvecs0.to(train_dev)
        Tdat0 = Tdat0.to(train_dev)
    Traindat = toda.TensorDataset(Tdat0[train_set], Tvecs0[train_set])
    Tbatchdat = toda.DataLoader(
        Traindat,
        batch_size=batch,
        shuffle=True,
    )

    losses = []
    counter = 0  # global counter
    deltat = 0
    # divstep = 0
    minlr = 1e-8
    batch_move = True
    layer_noise = 0
    for i in range(2):
        if i == 0:
            pass
        elif i == 1:
            layer_noise = 0
            batch = 10000
            batch_move = False
            print('final loop')
        params = myint.parameters()
        optim = torch.optim.Adam(params, lr=lr0)
        # optim = torch.optim.SGD(params, lr=lr0, nesterov=True, momentum=0.1)
        sched = getSchedOptim(optim)
        while True:
            tstart = time.time()
            counter += 1
            lossAccum00 = 0
            lossAccum_noised = 0
            if not batch_move:
                optim.zero_grad()
            for Tdat, Tvecs00 in Tbatchdat:
                # noise perturbed vectors
                Tvecs = Tvecs00 + torch.rand(
                    size=Tvecs00.size()).to(train_dev) * layer_noise
                if batch_on_dev:
                    Tdat = Tdat.to(train_dev)
                    Tvecs = Tvecs.to(train_dev)
                if batch_move:
                    optim.zero_grad()
                Rfinal_noised = myint(Tvecs) * tSD_0 + tD_0
                if layer_noise == 0:
                    Rfinal00 = Rfinal_noised
                else:
                    Rfinal00 = myint(Tvecs00) * tSD_0 + tD_0

                loss_noised = tofu.l1_loss(Rfinal_noised, Tdat) / spread0
                loss00 = tofu.l1_loss(Rfinal00, Tdat) / spread0
                if batch_move:
                    torch.autograd.backward(loss_noised)
                    optim.step()

                lossAccum00 += loss00 * len(Tdat) * npix
                lossAccum_noised += loss_noised * len(Tdat) * npix
            if not batch_move:
                torch.autograd.backward(lossAccum_noised / nspec / npix)
                optim.step()
            sched.step(lossAccum00)
            if validation:
                with torch.inference_mode():
                    val_loss = tofu.l1_loss(
                        myint(Tvecs0[validation_set]) * tSD_0 + tD_0,
                        Tdat0[validation_set]) / spread0
                    val_loss = val_loss.detach().cpu().numpy()
            else:
                val_loss = 0
            loss_V = lossAccum00.detach().cpu().numpy() / dats.size
            curlr = optim.param_groups[0]['lr']
            print('it %d loss %.5f' % (counter, loss_V), 'val %.5f' % val_loss,
                  'lr', curlr, 'time', deltat)
            # lastloss = loss_V
            losses.append(loss_V)
            if curlr < minlr:
                break

            if counter % 32 == 0 and counter > 0:
                print('saving')
                torch.save(myint.state_dict(), statefile_path)
            deltat = time.time() - tstart
    myint.pc_layer.weight.data[:, :] = tSD_0.view(
        npix, 1) * myint.pc_layer.weight.data[:, :]
    myint.pc_layer.bias.data[:] = tD_0[:] + myint.pc_layer.bias.data[:] * tSD_0

    torch.save(myint.state_dict(), finalfile_path)
    pred = get_predictions(myint, Tvecs0, train_dev, batch)
    cur_name = f'{directory}/pred_{setup}.psav'
    DD = {}
    DD['pred'] = pred,
    DD['vecs'] = vecs
    DD['dats'] = dats,
    DD['mapper'] = mapper
    DD['vecs_orig'] = vecs_orig
    serializer.save_dict_to_hdf5(cur_name, DD)

    if os.path.exists(statefile_path):
        os.unlink(statefile_path)
    import rvspecfit.nn.RVSInterpolator  # noqa

    ofname = f'{directory}/interp_{setup}.pkl'
    D = {
        'mapper': mapper,
        'parnames': parnames,
        'lam': lam,
        'log_spec': True,
        'logstep': True,
        'module': 'rvspecfit.nn.RVSInterpolator',
        'class_name': 'RVSInterpolator',
        'device': device_name,
        'class_kwargs': kwargs,
        'outside_class_name': 'OutsideInterpolator',
        'outside_kwargs': dict(pts=vecs),
        'nn_file': finalfile,
        'revision': revision,
        'git_rev': git_rev,
        'interpolation_type': 'generic'
    }
    serializer.save_dict_to_hdf5(ofname, D)


if __name__ == '__main__':
    main(sys.argv[1:])
