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
        tmpout = subprocess.Popen(
            'cd ' + os.path.dirname(__file__) +
            ' ; git log -n 1 --pretty=format:%H',
            shell=True,
            bufsize=80,
            stdout=subprocess.PIPE).stdout
        revision = tmpout.read().decode()[:6]
        return revision
    except:
        return ''


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


#VERSIONPIP = '0.0.1'
VERSIONPIP=read('version.txt').rstrip()
print (get_revision())
VERSION = VERSIONPIP + 'dev' + get_revision()

with open('py/rvspecfit/_version.py', 'w') as fp:
    print('version="%s"' % (VERSION), file=fp)

setup(
    name="rvspecfit",
    version=VERSION,
    author="Sergey Koposov",
    author_email="skoposov@cmu.edu",
    description=("Radial velocity code."),
    license="BSD",
    keywords="example documentation tutorial",
    url="http://github.com/segasai/rvspecfit",
    packages=['rvspecfit', 'rvspecfit/desi', 'rvspecfit/weave'],
    scripts=[fname for fname in glob.glob(os.path.join('bin', '*'))],
    package_dir={'': 'py/'},
    package_data={'rvspecfit': ['tests/']},
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
