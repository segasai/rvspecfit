import os
import time
import yaml
import logging
import numpy as np
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


class FileQueue:
    """Iterator that yields filenames from a list or a queue file.

    Can operate in three modes:
    - file_list: iterate over an in-memory list of filenames.
    - file_from (queue=False): read all filenames from a text file at init.
    - file_from (queue=True): treat the text file as a shared queue,
      atomically popping the first line on each call (file-lock based,
      safe for multiple processes on the same filesystem).
    """

    def __init__(self, file_list=None, file_from=None, queue=False):
        if file_list is not None:
            self.file_list = file_list
            self.file_from = None
            self.queue = False
        elif file_from is not None:
            if not queue:
                self.file_list = []
                with open(file_from, 'r') as fp:
                    for ll in fp:
                        self.file_list.append(ll.rstrip())
            else:
                self.file_list = None
                self.file_from = file_from
                self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        if self.file_list is not None:
            if len(self.file_list) > 0:
                return self.file_list.pop(0)
            else:
                raise StopIteration
        else:
            return self.read_next()

    def read_next(self):
        import socket
        lockname = self.file_from + '.%s.%d.lock' % (socket.gethostname(),
                                                     os.getpid())
        wait_time = 1
        max_waits = 1000
        for i in range(max_waits):
            try:
                os.rename(self.file_from, lockname)
            except FileNotFoundError:
                time.sleep(np.random.uniform(wait_time, 1.5 * wait_time))
                continue
            try:
                with open(lockname, 'r') as fp1:
                    ll = fp1.readlines()
                if len(ll) == 0:
                    raise StopIteration
                ret = ll[0].rstrip()
                with open(lockname, 'w') as fp1:
                    fp1.writelines(ll[1:])
                return ret
            finally:
                os.rename(lockname, self.file_from)

        logging.warning('Cannot read next file due to lock')
        raise StopIteration


class MPIFileQueue:
    """Distributes files across MPI ranks using a central server on rank 0.

    Architecture
    ------------
    Rank 0 runs two concurrent threads:
      - Main thread: acts as a local worker, pulling files via _pop_file().
      - Server thread: handles MPI requests from remote workers (ranks 1..N-1).
    Ranks 1..N-1 run a single thread that requests files via MPI send/recv.

    Lifecycle (state transitions)
    -----------------------------
    Rank 0 main thread:
      INIT       → constructs queue, starts server thread
      ITERATING  → calls __next__() → _pop_file() returns a filename
      EXHAUSTED  → _pop_file() returns None → StopIteration ends the for-loop
      SHUTDOWN   → caller invokes shutdown() → joins server thread → done

    Rank 0 server thread:
      WAITING    → blocked in probe(), waiting for any remote worker request
      SERVING    → received request, calls _pop_file(), sends result back:
                     • filename → worker continues, server goes back to WAITING
                     • None     → worker is done, decrement active_remote_workers
      FINISHED   → active_remote_workers == 0 → thread exits naturally

    Ranks 1..N-1 (remote workers):
      ITERATING  → calls __next__() → sends REQUEST_CMD → blocks in recv()
                     • receives filename → processes it → back to ITERATING
                     • receives None → StopIteration ends the for-loop
      SHUTDOWN   → caller invokes shutdown() → no-op on non-rank-0

    Termination guarantees
    ----------------------
    - Each remote worker receives exactly one None (its termination signal).
    - The server thread exits only after all N-1 remote workers have received
      None, so shutdown()/join() on rank 0 is guaranteed to complete.
    - The server thread is non-daemon, so even if shutdown() is not called
      (e.g. due to an exception), the process stays alive until the server
      thread finishes — preventing silent worker hangs.  The caller should
      still use try/finally to call shutdown() for clean exit.

    Edge cases
    ----------
    - 0 files: rank 0 gets StopIteration immediately.  Server thread still
      serves None to every remote worker, then exits.
    - 1 rank (no remote workers): server thread sees active_remote_workers=0,
      exits immediately.  Main thread processes all files locally.
    - Worker crash: the server thread stays blocked in probe() indefinitely
      because the crashed worker never sends its next request.  This is
      inherent to MPI — if a rank dies, MPI_Abort is the expected recovery.
    """

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
            # Must NOT be a daemon thread: if rank 0's main thread
            # finishes its own work and exits, a daemon thread would be
            # killed immediately, leaving remote workers hanging in recv().
            self.server_thread = threading.Thread(target=self._run_server, daemon=False)
            self.server_thread.start()

    def _pop_file(self):
        """Thread-safe retrieval of the next file from the queue.

        Returns the next filename, or None if all files have been consumed.
        Used by both the server thread (serving remote workers) and
        rank 0's main thread (acting as a local worker).
        """
        with self.lock:
            if self.index < self.num_files:
                val = self.file_list[self.index]
                self.index += 1
                return val
            return None

    def _run_server(self):
        # This thread ONLY runs on Rank 0 and handles requests from Ranks 1 to N
        active_remote_workers = self.size - 1
        
        while active_remote_workers > 0:
            status = self.MPI.Status()
            self.comm.probe(source=self.MPI.ANY_SOURCE, tag=self.MPI.ANY_TAG, status=status)
            request = self.comm.recv(source=status.source, tag=self.MPI.ANY_TAG)
            
            if request == self.REQUEST_CMD:
                file_to_send = self._pop_file()
                        
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
            val = self._pop_file()
            if val is not None:
                return val
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

    def shutdown(self):
        """Wait for the server thread to finish serving all remote workers.

        Must be called on all ranks after the iteration loop completes.
        On rank 0 this joins the server thread (which exits only after
        every remote worker has received its None termination message).
        On other ranks this is a no-op.
        """
        if self.rank == 0 and hasattr(self, 'server_thread'):
            self.server_thread.join()
    
