import numpy as np
import sys
import astropy.io.fits as pyfits

lamcen = 5000


def getspec(lam, teff, logg, alpha, met):
    w = 0.01 + (10 * logg / 5.)
    return (teff**4 * 1. / lam *
            (1 - min(1, np.exp(met)) * np.exp(-0.5 *
                                              (lam - lamcen)**2 / w**2)))


def make_grid(prefix, wavefile, nspec):
    S0 = np.random.get_state()
    np.random.seed(1)
    lam = np.linspace(4500, 5500, 50000)
    teffs = np.random.uniform(3000, 12000, nspec)
    fehs = np.random.uniform(-2, 0, nspec)
    alphas = np.random.uniform(0, 1, nspec)
    loggs = np.random.uniform(0, 5, nspec)
    loggs[0] = 4.5
    teffs[0] = 12000
    alphas[0] = 0
    fehs[0] = 0
    #     4.5, 12000, 0, 0
    for i in range(nspec):
        spec = getspec(lam, teffs[i], loggs[i], alphas[i], fehs[i])
        hdr = dict(PHXLOGG=loggs[i],
                   PHXALPHA=alphas[i],
                   PHXTEFF=teffs[i],
                   PHXM_H=fehs[i])
        fname = prefix + 'xx_%05d.fits' % i
        pyfits.writeto(fname, spec, pyfits.Header(hdr), overwrite=True)
    pyfits.writeto(wavefile, lam, overwrite=True)
    np.random.set_state(S0)


if __name__ == '__main__':
    make_grid(sys.argv[1], sys.argv[2], int(sys.argv[3]))
