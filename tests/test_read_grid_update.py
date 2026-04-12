import sqlite3
import pathlib

import numpy as np
import astropy.io.fits as pyfits

from rvspecfit import read_grid


def _write_template(path, teff, logg, feh, alpha):
    data = np.ones(16, dtype=np.float32)
    hdr = pyfits.Header()
    hdr['PHXTEFF'] = teff
    hdr['PHXLOGG'] = logg
    hdr['PHXM_H'] = feh
    hdr['PHXALPHA'] = alpha
    pyfits.writeto(path, data, hdr, overwrite=True)


def _count_rows(dbfile):
    with sqlite3.connect(dbfile) as conn:
        return conn.execute('select count(*) from files').fetchall()[0][0]


def test_makedb_update_adds_only_new_files(tmp_path):
    template_dir = tmp_path / 'templates'
    template_dir.mkdir()
    dbfile = tmp_path / 'files.db'
    prefix = str(template_dir) + '/'

    _write_template(template_dir / 't1.fits', 5000, 4.0, -1.0, 0.2)
    _write_template(template_dir / 't2.fits', 5100, 4.1, -0.9, 0.2)

    keywords = dict(teff='PHXTEFF', logg='PHXLOGG', feh='PHXM_H',
                    alpha='PHXALPHA')
    read_grid.makedb(prefix=prefix,
                     dbfile=str(dbfile),
                     keywords=keywords,
                     mask='*.fits',
                     name_metallicity='feh',
                     name_alpha='alpha')
    assert _count_rows(str(dbfile)) == 2

    _write_template(template_dir / 't3.fits', 5200, 4.2, -0.8, 0.3)
    read_grid.makedb(prefix=prefix,
                     dbfile=str(dbfile),
                     keywords=keywords,
                     mask='*.fits',
                     update=True,
                     name_metallicity='feh',
                     name_alpha='alpha')
    assert _count_rows(str(dbfile)) == 3

    # Re-running update with no new files should not duplicate rows.
    read_grid.makedb(prefix=prefix,
                     dbfile=str(dbfile),
                     keywords=keywords,
                     mask='*.fits',
                     update=True,
                     name_metallicity='feh',
                     name_alpha='alpha')
    assert _count_rows(str(dbfile)) == 3

