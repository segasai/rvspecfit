import os
import subprocess
import yaml
from rvspecfit import frozendict


def get_default_config():
    """Create a default parameter config ditctionary
    
    Returns
    -------
    ret: dict
        Dictionary with config params

"""
    D = {}
    # Configuration parameters, should be moved to the yaml file
    D['min_vel'] = -1000
    D['max_vel'] = 1000
    D['vel_step0'] = 5  # the starting step in velocities
    D['max_vsini'] = 500
    D['min_vsini'] = 1e-2
    D['min_vel_step'] = 0.2
    D['second_minimizer'] = True
    return D


def read_config(fname=None):
    """
    Read the configuration file and return the frozendict with it

    Parameters
    ----------

    fname: string, optional
        The path to the configuration file. If not given config.yaml in the
        current directory is used

    Returns
    -------
    config: frozendict
        The dictionary with the configuration from a file

    """
    if fname is None:
        fname = 'config.yaml'
    with open(fname) as fp:
        D = yaml.safe_load(fp)
        if D is None:
            D = {}
        D0 = get_default_config()
        for k in D0.keys():
            if k not in D:
                D[k] = D0[k]
        D['config_file_path'] = os.path.abspath(fname)
        return freezeDict(D)


def freezeDict(d):
    """ Take the input object and if it is a dictionary, 
    freeze it (i.e. return frozendict)
    If not, do nothing

    Parameters
    ----------

    d: dict
        Input dictionary

    Returns
    -------
    d: frozendict
        Frozen input dictionary

    """
    if isinstance(d, dict):
        d1 = {}
        for k, v in d.items():
            d1[k] = freezeDict(v)
        return frozendict.frozendict(d1)
    else:
        return d
