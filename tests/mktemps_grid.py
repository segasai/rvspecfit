import numpy as np
import sys
import astropy.io.fits as pyfits
import mktemps


def make_grid(prefix, wavefile):
    nt, nl, nf, na = 7, 7, 7, 7
    S0 = np.random.get_state()
    np.random.seed(1)
    lam = np.linspace(4500, 5500, 50000)
    teffs = np.linspace(mktemps.minteff, mktemps.maxteff, nt)
    fehs = np.linspace(-2, 0, nf)
    alphas = np.linspace(0, 1, na)
    loggs = np.linspace(0, 5, nl)
    i = 0
    for iit in range(nt):
        for iil in range(nl):
            for iif in range(nf):
                for iia in range(na):
                    spec = mktemps.getspec(lam, teffs[iit], loggs[iil],
                                           fehs[iif], alphas[iia])
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
