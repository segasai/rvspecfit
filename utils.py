import os
import subprocess
import yaml
import frozendict

def get_revision():
    """
    Get the git revision of the code

    Returns:
    --------
    revision : string
        The string with the git revision
    """
    try:
        fname = os.path.dirname(os.path.realpath(__file__))
        tmpout = subprocess.Popen(
            'cd ' + fname + ' ; git log -n 1 --pretty=format:%H -- make_nd.py',
            shell=True, bufsize=80, stdout=subprocess.PIPE).stdout
        revision = tmpout.read()
        return revision
    except:
        return ''

def read_config(fname=None):
    if fname is None:
        fname = 'config.yaml'
    with open(fname) as fp:
        return freezeDict(yaml.safe_load(fp))

def freezeDict(d):
    if isinstance(d, dict):
        d1 = {}
        for k, v in d.items():
            d1[k] = freezeDict(v)
        return frozendict.frozendict(d1)
    else:
        return d
