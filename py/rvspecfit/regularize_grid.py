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


def converter(path, opath):
    """
Read the input spectrum pickle file and convert it
into the file with gaps filled and smaller step sizes
    """
    newfehgrid = np.arange(-4, 1.25, .25)
    newalphagrid = np.arange(-0.4, 1.3, 0.2)

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

        RR = scipy.interpolate.Rbf(evalx1, evalx2, evalx3, evalx4, specs[xind,
                                                                         0])

        coeffs = scipy.linalg.solve(RR.A, specs[xind, :])
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
        r = RR._call_norm(newx, RR.xi)
        pred = np.dot(RR._function(r), coeffs)

        res_vec.append(newx0)
        res_spec.append(pred)

    res_vec = np.concatenate(res_vec, axis=1)
    res_spec = np.concatenate(res_spec, axis=0)
    dat['specs'] = res_spec
    dat['vec'] = res_vec

    with open(opath, 'wb') as fp:
        pickle.dump(dat, fp, protocol=4)


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input',
                        help='Input pickle',
                        type=str,
                        required=True)
    parser.add_argument('--output',
                        help='Input pickle',
                        type=str,
                        required=True)
    args = parser.parse_args(args)
    converter(args.input, args.output)


if __name__ == '__main__':
    main(sys.argv[1:])
