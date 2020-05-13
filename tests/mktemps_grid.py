import numpy as np
import sys
import astropy.io.fits as pyfits

lamcens = [5000.77, 5050.11, 5060.66]  # wavelength centers
lamamps = [1, 0.5, 0.2]  # line amplitudes
lammetamps = [1, 2, 0.4]  # the factor for the metallicity dependence


def getspec(lam, teff, logg, alpha, met, wresol=0):
    w = 0.01 + (10 * logg / 5.)
    cont = teff**4 * 1. / lam
    curw = np.sqrt(w**2 + wresol**2)
    lines = [(1 - min(1, np.exp(lammetamps[i] * met)) * lamamps[i] * w / curw *
              np.exp(-0.5 * (lam - lamcens[i])**2 / curw**2))
             for i in range(len(lamcens))]
    return np.prod(np.array(lines), axis=0) * cont


def make_grid(prefix, wavefile):
    nt, nl, nf, na = 7, 7, 7, 7
    S0 = np.random.get_state()
    np.random.seed(1)
    lam = np.linspace(4500, 5500, 50000)
    teffs = np.linspace(3000, 13000, nt)
    fehs = np.linspace(-2, 0, nf)
    alphas = np.linspace(0, 1, na)
    loggs = np.linspace(0, 5, nl)
    i = 0
    for iit in range(nt):
        for iil in range(nl):
            for iif in range(nf):
                for iia in range(na):
                    spec = getspec(lam, teffs[iit], loggs[iil], alphas[iia],
                                   fehs[iif])
                    hdr = dict(PHXLOGG=loggs[iil],
                               PHXALPHA=alphas[iia],
                               PHXTEFF=teffs[iit],
                               PHXM_H=fehs[iif])
                    fname = prefix + 'specs/xx_%05d.fits' % i
                    i += 1
                    pyfits.writeto(fname,
                                   spec,
                                   pyfits.Header(hdr),
                                   overwrite=True)
    pyfits.writeto(prefix + '/' + wavefile, lam, overwrite=True)
    np.random.set_state(S0)


if __name__ == '__main__':
    make_grid(sys.argv[1], sys.argv[2])
