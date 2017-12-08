import os
from setuptools import setup
import glob

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "rvspecfit",
    version = "0.0.1",
    author = "Sergey Koposov",
    author_email = "skoposov@cmu.edu",
    description = ("Radial velocity code."),
    license = "BSD",
    keywords = "example documentation tutorial",
    url = "http://github.com/segasai/rvspecfit",
    packages=['rvspecfit','rvspecfit/desi'],
    scripts = [fname for fname in glob.glob(os.path.join('bin', '*'))],
    package_dir={'':'py/'},
    package_data={'rvspecfit':['tests/']},
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
