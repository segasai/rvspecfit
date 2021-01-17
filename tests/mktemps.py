import numpy as np
import sys
import astropy.io.fits as pyfits

lamcens = np.r_[5000.77, 5050.11, 5060.66]  # wavelength centers
lamamps = np.r_[1, 0.5, 0.2]  # line amplitudes
lammetamps = np.r_[1, 2, 0.1]  # the factor for the metallicity dependence
lamteffamps = np.r_[.3, -.2, -.9]  # the factor for the teff dependence
lamwidths = np.r_[0.1, 0.1, 0.1]
minteff = 3000
maxteff = 12000


def getspec(lam, teff, logg, met, alpha, wresol=0, energy=True):
    w0 = 0.01 + (10 * logg / 5.)
    w0 = np.sqrt(w0**2 + lamwidths**2)
    cont = teff**4 * 1. / lam
    curw = np.sqrt(w0**2 + wresol**2)
    lines = []
    normteff = (teff - minteff) / (maxteff - minteff)
    curamps = np.clip(
        np.exp(lammetamps * met) *
        (1 + lamteffamps * normteff) * lamamps, 0, 1) * w0 / curw

    lines = (1 - curamps[None, :] *
             np.exp(-0.5 *
                    (lam[:, None] - lamcens[None, :])**2 / curw[None, :]**2))

    ret = np.prod(np.array(lines), axis=1) * cont
    if energy:
        ret = ret
    else:
        # in photons
        ret = ret * lam
    return ret


def make_grid(prefix, wavefile, nspec):
    S0 = np.random.get_state()
    np.random.seed(1)
    lam = np.linspace(4500, 5500, 50000)
    teffs = np.random.uniform(minteff, maxteff, nspec)
    fehs = np.random.uniform(-2, 0, nspec)
    alphas = np.random.uniform(0, 1, nspec)
    loggs = np.random.uniform(0, 5, nspec)
    loggs[0] = 4.5
    teffs[0] = 12000
    alphas[0] = 0
    fehs[0] = 0
    #     4.5, 12000, 0, 0
    for i in range(nspec):
        spec = getspec(lam, teffs[i], loggs[i], fehs[i], alphas[i])
        hdr = dict(PHXLOGG=loggs[i],
                   PHXALPHA=alphas[i],
                   PHXTEFF=teffs[i],
                   PHXM_H=fehs[i])
        fname = prefix + 'specs/xx_%05d.fits' % i
        pyfits.writeto(fname, spec, pyfits.Header(hdr), overwrite=True)
    pyfits.writeto(prefix + '/' + wavefile, lam, overwrite=True)
    np.random.set_state(S0)


if __name__ == '__main__':
    make_grid(sys.argv[1], sys.argv[2], int(sys.argv[3]))
