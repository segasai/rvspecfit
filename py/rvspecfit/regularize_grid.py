import pickle
import sys
import argparse
import scipy.stats
import scipy.interpolate
import numpy as np


def findbestoverlaps(x, intervals):
    """find the interval where the value is closer to the center
I.e. given intervals [0,10],[1,11],[2,12],[3,13],[4,14],[5,15],[6,16]
and value of 8 it return [3,13]
"""
    bestx = np.zeros(len(x)) + 1e10
    bestid = np.zeros(len(x), dtype=int)
    for i, curi in enumerate(intervals):
        curv = (x - curi[0]) * (x - curi[-1])
        xind = (bestx > curv)
        bestid[xind] = i
        bestx[xind] = curv[xind]
    return bestid


def check_holes_2d(x, y):
    # check that the x,y locations form a grid without holes
    ux, ix = np.unique(x, return_inverse=True)
    uy, iy = np.unique(y, return_inverse=True)
    nx = len(ux)
    ny = len(uy)
    arr = np.zeros((len(ux), len(uy)))
    arr[ix, iy] = 1
    xholes, yholes = np.nonzero(1 - arr)
    for cx, cy in zip(xholes, yholes):
        if cx == 0 or cx == nx - 1 or cy == 0 or cy == ny - 1:
            continue
        cnt = arr[cx - 1, cy - 1] + arr[cx + 1, cy + 1] + arr[cx - 1, cy +
                                                              1] + arr[cx + 1,
                                                                       cy - 1]
        if cnt == 3 or cnt == 4:
            raise Exception('the grid has holes')


def converter(path,
              opath,
              smooth=0,
              min_feh=None,
              max_feh=None,
              step_feh=None,
              min_alpha=None,
              max_alpha=None,
              step_alpha=None):
    """
Read the input spectrum pickle file and convert it
into the file with gaps filled and smaller step sizes
    """
    # I'm adding half a step to ensure that max value is included
    newfehgrid = np.arange(min_feh, max_feh + step_feh / 2., step_feh)
    newalphagrid = np.arange(min_alpha, max_alpha + step_alpha / 2.,
                             step_alpha)

    dat = pickle.load(open(path, 'rb'))

    vec = dat['vec']
    specs = dat['specs']
    teff, logg, feh, alpha = vec

    uteff, teffid = np.unique(teff, return_inverse=True)
    ulogg, loggid = np.unique(logg, return_inverse=True)
    ufeh, fehid = np.unique(feh, return_inverse=True)
    ualpha, alphaid = np.unique(alpha, return_inverse=True)

    # important that I don't use the s=0
    mappers = [
        scipy.interpolate.UnivariateSpline(_, np.arange(len(_)))
        for _ in [uteff, ulogg, ufeh, ualpha]
    ]

    vec_map = [mappers[_](dat['vec'][_]) for _ in range(4)]
    # ranks smoothly transformed into rank grid that I use to interpolate

    teff_grid2d, logg_grid2d = np.array(list(set(zip(teff, logg)))).T
    # these are unique teff and logg locations of grid points in 2d
    # I assume they have no holes (I checked)
    check_holes_2d(teff, logg)

    # these a rank transformed teff, logg points
    teff_grid2d_map, logg_grid2d_map = [
        mappers[_](__) for _, __ in [(0, teff_grid2d), (1, logg_grid2d)]
    ]

    teff_grid2d_rank = np.digitize(teff_grid2d, uteff) - 1

    width = 12

    edges = np.arange(0, len(uteff) - width)  # inclusive edges

    intervals = np.array([(_, _ + width) for _ in edges])

    bestinterval = findbestoverlaps(teff_grid2d_rank, intervals)

    res_vec = []
    res_spec = []

    for ii, (cure1, cure2) in enumerate(intervals):
        print('cure', cure1, cure2)
        xind = (teffid >= cure1) & (teffid <= cure2)
        print('trainsub', xind.sum())

        evalx1, evalx2, evalx3, evalx4 = (vec_map[0][xind], vec_map[1][xind],
                                          vec_map[2][xind], vec_map[3][xind])

        RR = scipy.interpolate.RBFInterpolator(np.array(
            [evalx1, evalx2, evalx3, evalx4]).T,
                                               specs[xind, :],
                                               smoothing=smooth)

        xind1 = bestinterval == ii

        x1, x2, x3, x4 = (teff_grid2d[xind1], logg_grid2d[xind1], newfehgrid,
                          newalphagrid)
        x1, x2, x3, x4 = x1[:, None, None], x2[:, None,
                                               None], x3[None, :,
                                                         None], x4[None,
                                                                   None, :]
        x1, x2, x3, x4 = [(_ + (x1 + x2 + x3 + x4) * 0).flatten()
                          for _ in [x1, x2, x3, x4]]
        # these arrays of points we predicting on

        print('predsub', len(x1))
        newx0 = np.array([x1, x2, x3, x4])
        newx = np.array([mappers[_]([x1, x2, x3, x4][_]) for _ in range(4)])
        pred = RR(newx.T)

        res_vec.append(newx0)
        res_spec.append(pred)

    res_vec = np.concatenate(res_vec, axis=1)
    res_spec = np.concatenate(res_spec, axis=0)
    dat['specs'] = res_spec
    dat['vec'] = res_vec

    with open(opath, 'wb') as fp:
        pickle.dump(dat, fp, protocol=4)


def check_scipy_version():
    import scipy.version
    if [int(_) for _ in scipy.version.version.split('.')] < [1, 9, 0]:
        raise RuntimeError('scipy 1.9.0+ is required due to Rbf changes')


def main(args):
    parser = argparse.ArgumentParser()
    check_scipy_version()
    parser.add_argument('--input',
                        help='Input pickle',
                        type=str,
                        required=True)
    parser.add_argument('--output',
                        help='Input pickle',
                        type=str,
                        required=True)

    parser.add_argument('--max_feh',
                        help='Max feh',
                        type=float,
                        required=False,
                        default=1.2)
    parser.add_argument('--min_feh',
                        help='Min feh',
                        type=float,
                        required=False,
                        default=-4)
    parser.add_argument('--max_alpha',
                        help='Max alpha',
                        type=float,
                        required=False,
                        default=1.2)
    parser.add_argument('--min_alpha',
                        help='Min alpha',
                        type=float,
                        required=False,
                        default=-.4)
    parser.add_argument('--step_feh',
                        help='step feh',
                        type=float,
                        required=False,
                        default=.25)
    parser.add_argument('--step_alpha',
                        help='step alpha',
                        type=float,
                        required=False,
                        default=.2)
    parser.add_argument('--smooth',
                        help='smoothing Parameter',
                        type=float,
                        default=0.)
    args = parser.parse_args(args)
    converter(
        args.input,
        args.output,
        smooth=args.smooth,
        min_feh=args.min_feh,
        max_feh=args.max_feh,
        step_feh=args.step_feh,
        min_alpha=args.min_alpha,
        max_alpha=args.max_alpha,
        step_alpha=args.step_alpha,
    )


if __name__ == '__main__':
    main(sys.argv[1:])
