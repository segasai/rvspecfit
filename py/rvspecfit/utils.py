import os
import yaml
import logging
from rvspecfit import frozendict
import threading

def get_default_config():
    """Create a default parameter config dictionary

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


class MPIFileQueue:
    def __init__(self, file_list=None):
        from mpi4py import MPI
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.file_list = file_list if self.rank == 0 else None
        self.REQUEST_CMD = 'file'
        
        if self.rank == 0:
            self.index = 0
            self.num_files = len(self.file_list)
            # Lock ensures the thread and the main process don't grab the same file
            self.lock = threading.Lock() 
            
            # Start the background server immediately upon instantiation
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()

    def _run_server(self):
        # This thread ONLY runs on Rank 0 and handles requests from Ranks 1 to N
        active_remote_workers = self.size - 1
        
        while active_remote_workers > 0:
            status = self.MPI.Status()
            self.comm.probe(source=self.MPI.ANY_SOURCE, tag=self.MPI.ANY_TAG, status=status)
            request = self.comm.recv(source=status.source, tag=self.MPI.ANY_TAG)
            
            if request == self.REQUEST_CMD:
                file_to_send = None
                
                # Safely grab the next file
                with self.lock:
                    if self.index < self.num_files:
                        file_to_send = self.file_list[self.index]
                        self.index += 1
                        
                # Send the file (or None if empty) to the requesting rank
                self.comm.send(file_to_send, dest=status.source)
                
                # If we sent None, that remote worker is done
                if file_to_send is None:
                    active_remote_workers -= 1
            else:
                raise RuntimeError('Unsupported message')

    def __next__(self):
        if self.rank == 0:
            # Rank 0's main thread acts as a worker, pulling locally.
            # No MPI overhead and no risk of self-messaging deadlocks.
            with self.lock:
                if self.index < self.num_files:
                    val = self.file_list[self.index]
                    self.index += 1
                    return val
                else:
                    raise StopIteration
        else:
            # Ranks > 0 request via MPI
            self.comm.send(self.REQUEST_CMD, dest=0)
            file_name = self.comm.recv(source=0, tag=self.MPI.ANY_TAG)
            if file_name is None:
                raise StopIteration
            return file_name

    def __iter__(self):
        return self
    
