from __future__ import print_function
import os
from setuptools import setup
import glob
import subprocess


def get_revision():
    """
    Get the git revision of the code

    Returns:
    --------
    revision : string
        The string with the git revision
    """
    try:
        tmpout = subprocess.Popen('cd ' + os.path.dirname(__file__) +
                                  ' ; git log -n 1 --pretty=format:%H',
                                  shell=True,
                                  bufsize=80,
                                  stdout=subprocess.PIPE).stdout
        revision = tmpout.read().decode()[:6]
        return revision
    except:  # noqa
        return ''


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


VERSIONPIP = read('version.txt').rstrip()
print(get_revision())
VERSION = VERSIONPIP + '+dev' + get_revision()

with open('py/rvspecfit/_version.py', 'w') as fp:
    print('version="%s"' % (VERSION), file=fp)

setup(
    name="rvspecfit",
    version=VERSION,
    author="Sergey Koposov",
    author_email="skoposov@ed.ac.uk",
    description=("Radial velocity and stellar parameter measurement code."),
    license="BSD",
    keywords="stellar spectra radial velocity",
    url="http://github.com/segasai/rvspecfit",
    packages=[
        'rvspecfit', 'rvspecfit/desi', 'rvspecfit/weave', 'rvspecfit/nn'
    ],
    scripts=[fname for fname in glob.glob(os.path.join('bin', '*'))],
    zip_safe=False,
    package_dir={'': 'py/'},
    package_data={'rvspecfit': ['tests/']},
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 5 - Production/Stable"
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering :: Astronomy"
        "License :: OSI Approved :: BSD License",
    ],
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=["py/rvspecfit/ffibuilder.py:ffibuilder"],
    install_requires=["cffi", "astropy", "pyyaml", "numpy", "scipy", "h5py"],
)
