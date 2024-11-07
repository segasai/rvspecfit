import os
import yaml
import logging
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
    D['template_lib'] = 'templ_data/'
    return D


def read_config(fname=None, override_options=None):
    """
    Read the configuration file and return the frozendict with it

    Parameters
    ----------

    fname: string, optional
        The path to the configuration file. If not given config.yaml in the
        current directory is used
    override_options: dictionary, optional
        Update the options
    Returns
    -------
    config: frozendict
        The dictionary with the configuration from a file

    """
    fname_specified = fname is not None
    if fname is None:
        fname = 'config.yaml'
    if os.path.exists(fname):
        fp = open(fname, 'r')
        D = yaml.safe_load(fp)
        if D is None:
            D = {}
            logging.warning('Configuration file config.yaml is empty. ' +
                            'Using default settings')
        fp.close()
    else:
        if fname_specified:
            raise RuntimeError(f"Configuration file '{fname}' not found.")
        else:
            logging.warning(f"Configuration file '{fname}' not found. "
                            "Using default settings")
            D = {}
    D0 = get_default_config()
    for k in D0.keys():
        if k not in D:
            logging.debug(
                'Keyword %s not found in configuration file. ' +
                'Using default value %s', k, D0[k])
            D[k] = D0[k]
    D['config_file_path'] = os.path.abspath(fname)
    if override_options is not None:
        for k, v in override_options.items():
            if k in D and v != D[k]:
                logging.warning(f'Provided option {k} overrides the value in '
                                'the configuration file')
            D[k] = v
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
    if isinstance(d, list):
        return tuple(d)
    else:
        return d
