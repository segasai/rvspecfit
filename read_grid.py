import glob
import sqlite3
import astropy.io.fits as pyfits
import scipy.stats
import scipy.sparse
import sqlite3
import numpy as np
import astropy.wcs as pywcs
import argparse


def makedb(prefix='/physics2/skoposov/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/',
           dbfile='files.db'):
    """ Create an sqlite database of the templates """
    DB = sqlite3.connect(dbfile)
    id = 0
    DB.execute(
        'CREATE TABLE files (filename varchar, teff real, logg real, met real, alpha real, id int);')

    for f in sorted(glob.glob(prefix + '*/*fits')):
        hdr = pyfits.getheader(f)
        teff = hdr['PHXTEFF']
        logg = float(hdr['PHXLOGG'])
        alpha = float(hdr['PHXALPHA'])
        met = float(hdr['PHXM_H'])

        DB.execute('insert into files  values (?, ? , ? , ? , ?, ? )',
                        (f.replace(prefix, ''), teff, logg, met, alpha, id))
        id += 1
    DB.commit()

def get_spec(logg, temp, met, alpha,
             dbfile='/tmp/files.db',
             prefix='/physics2/skoposov/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/',
             wavefile='/physics2/skoposov/phoenix.astro.physik.uni-goettingen.de/v2.0/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'):
    """ returns individual spectra
    > lam,spec=read_grid.get_spec(1,5250,-1,0.4)
    """
    deltalogg = 0.01
    deltatemp = 1
    deltaalpha = 0.01
    deltamet = 0.01

    query = '''select filename from files where
		teff between %f and %f and
		logg between %f and %f and
		alpha between %f and %f and
		met between %f and %f ''' % (
        temp - deltatemp, temp + deltatemp,
        logg - deltalogg, logg + deltalogg,
        alpha - deltaalpha, alpha + deltaalpha,
        met - deltamet, met + deltamet)

    conn = sqlite3.connect(dbfile)
    cur = conn.cursor()
    cur.execute(query)
    fnames = cur.fetchall()
    if len(fnames) > 1:
        print('Warning: More than 1 file returned', file=sys.stderr)
    if len(fnames) == 0:
        raise Exception('No spectra found')

    dat, hdr = pyfits.getdata(prefix + '/' + fnames[0][0], header=True)
    wc = pywcs.WCS(hdr)
    speclen = len(dat)
    lams = np.arange(speclen) * 1.
    lams = pyfits.getdata(wavefile)
    # assert(hdr['CTYPE1']=='AWAV-LOG')
    # lams=np.exp(wc.all_pix2sky(lams,lams*0,0)[0])
    print('Using', fnames[0][0])
    return lams, dat


def make_rebinner(lam00, lam, resolution, toair=True):
    """ Make the sparse matrix that convolves a given spectrum to
    a given resolution and new wavelength grid
    """
    if toair:
        lam0 = lam00 / (1.0 + 2.735182E-4 + 131.4182 /
                        lam00**2 + 2.76249E8 / lam00**4)
    else:
        lam0 = lam00

    resol0 = 100000
    assert (resolution < resol0)

    fwhms = lam / resolution
    fwhms0 = lam / resol0

    sigs = (fwhms**2 - fwhms0**2)**.5 / 2.35
    thresh = 5  # 5 sigma
    l0 = len(lam0)
    l = len(lam)
    xs = []
    ys = []
    vals = []
    for i in range(len(lam)):

        curlam = lam[i]
        cursig = sigs[i]
        curl0 = curlam - thresh * cursig
        curl1 = curlam + thresh * cursig

        left = np.searchsorted(lam0, curl0)
        right = np.searchsorted(lam0, curl1)

        curx = np.arange(left, right + 1)
        #curvals = scipy.stats.norm.pdf(lam0[curx], curlam, cursig)
        li = lam0[curx]
        li_p = lam0[curx + 1]
        C = scipy.special.erf((li_p - curlam) / np.sqrt(2) / cursig) -\
            scipy.special.erf((li - curlam) / np.sqrt(2) / cursig)
        D = np.exp(-0.5 * ((li - curlam) / cursig)**2) - np.exp(-0.5 *
                                                                ((li_p - curlam) / cursig)**2)

        curvals1 = (C * np.sqrt(2 * np.pi) / 2 * li *
                    (li_p - curlam) - cursig * li * D) / (li_p - li)
        curvals2 = (C * np.sqrt(2 * np.pi) / 2 * li_p *
                    (curlam - li) + cursig * li_p * D) / (li_p - li)
        ys.append(i + curx * 0)
        xs.append(curx)
        vals.append(curvals1)

        ys.append(i + curx * 0)
        xs.append(curx + 1)
        vals.append(curvals2)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    vals = np.concatenate(vals)

    mat = scipy.sparse.coo_matrix(
        (vals, (xs, ys)), shape=(len(lam0), len(lam)))
    mat = mat.tocsc()
    return mat


def apply_rebinner(mat, spec0):
    ret = np.array(spec0 * mat)
    return ret


def rebin(lam0, spec0, newlam, resolution):
    """rebin a given spectrum lam0, spec0 to the new wavelenght and new resolution
    > lam,spec=read_grid.get_spec(1,5250,-1,0.4)
    > newlam = np.linspace(4000,9000,4000)
    > newspec=read_grid.rebin(lam, spec, newlam, 1800)
    """
    lam, spec = get_spec(1, 5000, -1, 0.2)

    mat = make_rebinner(lam0, newlam, resolution)
    ret = apply_rebinner(mat, spec0)
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='./', dest='prefix')
    parser.add_argument('--templdb', type=str, default='files.db')
    args = parser.parse_args()
    makedb(args.prefix, dbfile=args.templdb)
