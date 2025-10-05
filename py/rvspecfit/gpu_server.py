"""
GPU Server for batched spectrum operations

This module implements a dedicated GPU worker process that handles
batched template evaluation, convolution, and chi-square computation,
minimizing data transfer and queue overhead.
"""

import multiprocessing as mp
import numpy as np
import time
import logging
from queue import Empty


class GPUSpectrumServer:
    """
    Server that runs in a dedicated process and handles GPU spectrum operations
    including template evaluation, convolution, and chi-square computation
    """

    def __init__(self, config, device_id=0, batch_timeout=0.002, max_batch_size=128, response_dict=None):
        """
        Parameters
        ----------
        config : dict
            Configuration dictionary with template_lib and other settings
        device_id : int
            GPU device ID to use
        batch_timeout : float
            Maximum time to wait for batch to fill (seconds)
        max_batch_size : int
            Maximum batch size for NN evaluation
        response_dict : Manager.dict
            Shared dictionary for responses
        """
        self.config = config
        self.device_id = device_id
        self.batch_timeout = batch_timeout
        self.max_batch_size = max_batch_size
        self.response_dict = response_dict

        # Queues for communication
        self.request_queue = mp.Queue()
        self.stats_queue = mp.Queue()  # For batch statistics

        # Process handle
        self.process = None
        self.shutdown_event = mp.Event()

    def start(self):
        """Start the GPU server process"""
        self.process = mp.Process(target=self._run_server, daemon=False)
        self.process.start()
        logging.info(f'GPU server started on device {self.device_id} (PID {self.process.pid})')

    def get_stats(self):
        """Get batching statistics from the GPU server"""
        stats = []
        while not self.stats_queue.empty():
            try:
                stats.append(self.stats_queue.get_nowait())
            except:
                break
        return stats

    def stop(self):
        """Stop the GPU server process"""
        self.shutdown_event.set()
        if self.process is not None:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()

        # Print statistics
        stats = self.get_stats()
        logging.info(f'GPU server stop: collected {len(stats)} stat entries')
        if stats:
            batch_sizes = [s['batch_size'] for s in stats]
            total_requests = stats[-1]['total'] if stats else 0
            import numpy as np
            logging.info(f'GPU server processed {total_requests} requests in {len(batch_sizes)} batches')
            logging.info(f'  Batch size: avg={np.mean(batch_sizes):.2f}, median={np.median(batch_sizes):.0f}, max={np.max(batch_sizes)}')
            logging.info(f'  Batches with size>1: {sum(1 for b in batch_sizes if b > 1)} ({sum(1 for b in batch_sizes if b > 1)/len(batch_sizes)*100:.1f}%)')
        else:
            logging.info('GPU server: no statistics collected (queue was empty)')

        logging.info('GPU server stopped')

    def _run_server(self):
        """Main server loop (runs in separate process)"""
        import os
        os.environ['RVS_NN_DEVICE'] = f'cuda:{self.device_id}'

        # DEBUG: File logging
        with open('/tmp/gpu_server_process.log', 'a') as f:
            f.write(f'GPU server process started, PID={os.getpid()}\n')

        # Import here to ensure GPU initialization in correct process
        from rvspecfit import spec_inter

        with open('/tmp/gpu_server_process.log', 'a') as f:
            f.write('Imported spec_inter\n')

        logging.info(f'GPU server initializing interpolators on cuda:{self.device_id}')

        # Pre-load interpolators for all setups
        interpolators = {}
        setups = ['desi_b', 'desi_r', 'desi_z']
        for setup in setups:
            try:
                with open('/tmp/gpu_server_process.log', 'a') as f:
                    f.write(f'Loading interpolator for {setup}\n')
                interpolators[setup] = spec_inter.getInterpolator(setup, self.config, warmup_cache=False)
                logging.info(f'GPU server loaded interpolator for {setup}')
                with open('/tmp/gpu_server_process.log', 'a') as f:
                    f.write(f'Loaded interpolator for {setup}\n')
            except Exception as e:
                logging.error(f'Failed to load interpolator for {setup}: {e}')
                with open('/tmp/gpu_server_process.log', 'a') as f:
                    f.write(f'ERROR loading {setup}: {e}\n')

        # Batch processing state
        pending_requests = []
        last_batch_time = time.time()

        with open('/tmp/gpu_server_process.log', 'a') as f:
            f.write('GPU server ready, entering main loop\n')
        logging.info('GPU server ready, entering main loop')

        # Statistics
        batch_sizes = []
        total_requests = 0

        while not self.shutdown_event.is_set():
            try:
                # Collect requests until batch is full or timeout
                while len(pending_requests) < self.max_batch_size:
                    timeout = max(0.001, self.batch_timeout - (time.time() - last_batch_time))
                    try:
                        request = self.request_queue.get(timeout=timeout)
                        if request is None:  # Shutdown signal
                            break
                        pending_requests.append(request)
                        with open('/tmp/gpu_server_process.log', 'a') as f:
                            f.write(f'Received request: {request.get("operation", "unknown")}\n')
                    except Empty:
                        break

                # Process batch if we have requests
                if pending_requests:
                    batch_size = len(pending_requests)
                    batch_sizes.append(batch_size)
                    total_requests += batch_size

                    # Send batch size to main process
                    try:
                        self.stats_queue.put({'batch_size': batch_size, 'total': total_requests}, block=False)
                    except:
                        pass  # Queue full, ignore

                    # Also log to file for debugging
                    with open('/tmp/gpu_server_batches.log', 'a') as f:
                        f.write(f'{batch_size}\n')

                    self._process_batch(pending_requests, interpolators)
                    pending_requests = []
                    last_batch_time = time.time()

            except Exception as e:
                logging.error(f'GPU server error: {e}')
                import traceback
                traceback.print_exc()

        # Print final statistics
        if batch_sizes:
            import numpy as np
            logging.info(f'GPU server final stats: {total_requests} total requests in {len(batch_sizes)} batches')
            logging.info(f'  Batch size: avg={np.mean(batch_sizes):.2f}, median={np.median(batch_sizes):.0f}, max={np.max(batch_sizes)}')
            logging.info(f'  Batches with size>1: {sum(1 for b in batch_sizes if b > 1)} ({sum(1 for b in batch_sizes if b > 1)/len(batch_sizes)*100:.1f}%)')

        logging.info('GPU server shutting down')

    def _process_batch(self, requests, interpolators):
        """Process a batch of requests"""

        # Group by operation type and setup
        by_operation = {}
        for req in requests:
            op_type = req.get('operation', 'eval_template')
            if op_type not in by_operation:
                by_operation[op_type] = []
            by_operation[op_type].append(req)

        # Process each operation type
        for op_type, op_requests in by_operation.items():
            if op_type == 'eval_template':
                self._process_template_batch(op_requests, interpolators)
            elif op_type == 'eval_chisq':
                self._process_chisq_batch(op_requests)
            elif op_type == 'convolve_vsini':
                self._process_convolve_batch(op_requests)
            else:
                logging.warning(f'Unknown operation type: {op_type}')

    def _process_template_batch(self, requests, interpolators):
        """Process template evaluation requests"""
        # Group by setup
        by_setup = {}
        for req in requests:
            setup = req['setup']
            if setup not in by_setup:
                by_setup[setup] = []
            by_setup[setup].append(req)

        # Process each setup's batch
        for setup, setup_requests in by_setup.items():
            if setup not in interpolators:
                logging.warning(f'No interpolator for {setup}')
                continue

            interp = interpolators[setup]

            # Extract parameters
            params_list = [req['params'] for req in setup_requests]

            # Batch evaluate
            try:
                if hasattr(interp, 'interper') and hasattr(interp.interper, '__call__'):
                    # Evaluate each individually for now (can optimize later)
                    results = []
                    for params in params_list:
                        spec = interp.eval(params)
                        results.append({
                            'spec': spec,
                            'lam': interp.lam,
                            'outside': float(interp.outsideFlag(params)),
                            'log_step': interp.log_step
                        })

                    # Send responses back to workers via shared dict
                    for req, result in zip(setup_requests, results):
                        request_id = req['request_id']
                        self.response_dict[request_id] = result

            except Exception as e:
                logging.error(f'Batch evaluation error for {setup}: {e}')
                # Send error responses
                for req in setup_requests:
                    self.response_dict[req['request_id']] = {'error': str(e)}

    def _process_chisq_batch(self, requests):
        """Process chi-square computation requests in batch"""
        from rvspecfit import spec_fit_gpu

        # Collect all specs and templates
        specs = []
        templs = []
        especs = []

        for req in requests:
            specs.append(req['spec'])
            templs.append(req['templ'])
            especs.append(req.get('espec', None))

        # All spectra use the same polynomial basis
        polys_arr = requests[0]['poly']

        try:
            # Batch compute chi-squares on GPU
            specs_arr = np.array(specs)
            templs_arr = np.array(templs)
            especs_arr = np.array(especs) if especs[0] is not None else None

            chisqs, coeffs = spec_fit_gpu.get_chisq0_batch_gpu(
                specs_arr, templs_arr, polys_arr, especs_arr,
                device_id=self.device_id
            )

            # Send responses via shared dict
            for req, chisq, coeff in zip(requests, chisqs, coeffs):
                self.response_dict[req['request_id']] = {
                    'chisq': float(chisq),
                    'coeffs': coeff
                }

        except Exception as e:
            logging.error(f'Batch chi-square error: {e}')
            for req in requests:
                self.response_dict[req['request_id']] = {'error': str(e)}

    def _process_convolve_batch(self, requests):
        """Process vsini convolution requests in batch"""
        from rvspecfit import spec_fit_gpu

        # Group by wavelength grid (must be same for batching)
        by_wavelength = {}
        for req in requests:
            lam_key = tuple(req['lam'][:10])  # Use first 10 points as key
            if lam_key not in by_wavelength:
                by_wavelength[lam_key] = []
            by_wavelength[lam_key].append(req)

        for lam_key, batch_requests in by_wavelength.items():
            try:
                lam_templ = batch_requests[0]['lam']
                templs = np.array([req['templ'] for req in batch_requests])
                vsinis = np.array([req['vsini'] for req in batch_requests])

                # Batch convolve on GPU
                convolved = spec_fit_gpu.convolve_vsini_batch_gpu(
                    lam_templ, templs, vsinis, device_id=self.device_id
                )

                # Send responses via shared dict
                for req, conv_spec in zip(batch_requests, convolved):
                    self.response_dict[req['request_id']] = {
                        'spec': conv_spec
                    }

            except Exception as e:
                logging.error(f'Batch convolution error: {e}')
                for req in batch_requests:
                    self.response_dict[req['request_id']] = {'error': str(e)}

    def eval_template(self, setup, params, worker_id=None):
        """
        Request template evaluation from GPU server

        Parameters
        ----------
        setup : str
            Setup name (e.g., 'desi_b')
        params : tuple or array
            Atmospheric parameters (teff, logg, feh, alpha)
        worker_id : int
            Worker ID for response routing

        Returns
        -------
        dict with keys: 'spec', 'lam', 'outside', 'log_step'
        """
        response_queue = mp.Queue()
        request = {
            'operation': 'eval_template',
            'setup': setup,
            'params': tuple(params),
            'response_queue': response_queue,
            'worker_id': worker_id
        }
        self.request_queue.put(request)

        try:
            result = response_queue.get(timeout=30)
            if 'error' in result:
                raise RuntimeError(f"GPU server error: {result['error']}")
            return result
        except Empty:
            raise TimeoutError('GPU server did not respond in time')

    def eval_chisq(self, spec, templ, poly, espec=None):
        """
        Request chi-square computation from GPU server

        Parameters
        ----------
        spec : array
            Observed spectrum
        templ : array
            Template spectrum
        poly : array
            Polynomial basis
        espec : array, optional
            Error spectrum

        Returns
        -------
        dict with keys: 'chisq', 'coeffs'
        """
        response_queue = mp.Queue()
        request = {
            'operation': 'eval_chisq',
            'spec': spec,
            'templ': templ,
            'poly': poly,
            'espec': espec,
            'response_queue': response_queue
        }
        self.request_queue.put(request)

        try:
            result = response_queue.get(timeout=30)
            if 'error' in result:
                raise RuntimeError(f"GPU server error: {result['error']}")
            return result
        except Empty:
            raise TimeoutError('GPU server did not respond in time')

    def convolve_vsini(self, lam, templ, vsini):
        """
        Request vsini convolution from GPU server

        Parameters
        ----------
        lam : array
            Wavelength array
        templ : array
            Template spectrum
        vsini : float
            Rotation velocity

        Returns
        -------
        dict with key: 'spec' (convolved spectrum)
        """
        response_queue = mp.Queue()
        request = {
            'operation': 'convolve_vsini',
            'lam': lam,
            'templ': templ,
            'vsini': vsini,
            'response_queue': response_queue
        }
        self.request_queue.put(request)

        try:
            result = response_queue.get(timeout=30)
            if 'error' in result:
                raise RuntimeError(f"GPU server error: {result['error']}")
            return result
        except Empty:
            raise TimeoutError('GPU server did not respond in time')


# Global server instance
_gpu_server = None
_gpu_server_queues = None  # Store queues for worker access
_gpu_response_manager = None  # Shared Manager for response queues


def start_gpu_server(config, device_id=0, batch_timeout=0.002, max_batch_size=128):
    """Start the global GPU server and return it with queues"""
    global _gpu_server, _gpu_server_queues, _gpu_response_manager
    if _gpu_server is None:
        # Start shared Manager for response dict (simpler than Queue per request)
        from multiprocessing import managers
        _gpu_response_manager = managers.SyncManager()
        _gpu_response_manager.start()
        # Create a shared dict for responses indexed by request_id
        response_dict = _gpu_response_manager.dict()

        _gpu_server = GPUSpectrumServer(config, device_id, batch_timeout, max_batch_size, response_dict=response_dict)
        _gpu_server.start()
        # Store queues globally so worker processes can access them
        _gpu_server_queues = {
            'request_queue': _gpu_server.request_queue,
            'stats_queue': _gpu_server.stats_queue,
            'response_dict': response_dict
        }
    return _gpu_server


def _worker_init(gpu_queues):
    """Initialize worker process with GPU server queues"""
    global _gpu_server_queues
    _gpu_server_queues = gpu_queues


def stop_gpu_server():
    """Stop the global GPU server"""
    global _gpu_server, _gpu_response_manager
    if _gpu_server is not None:
        _gpu_server.stop()
        _gpu_server = None
    if _gpu_response_manager is not None:
        _gpu_response_manager.shutdown()
        _gpu_response_manager = None


def get_gpu_server():
    """Get the global GPU server instance or create a client stub for workers"""
    global _gpu_server, _gpu_server_queues

    # If we have the server object, return it
    if _gpu_server is not None:
        return _gpu_server

    # If we have queues but no server (we're in a worker process),
    # create a client stub that uses the shared queues
    if _gpu_server_queues is not None:
        # Create a minimal client object that just has the request methods
        class GPUServerClient:
            def __init__(self, queues):
                self.request_queue = queues['request_queue']
                self.response_dict = queues['response_dict']
                self.request_counter = 0

            def eval_template(self, setup, params, worker_id=None):
                import os
                import time
                # Generate unique request ID
                request_id = f"{os.getpid()}_{self.request_counter}"
                self.request_counter += 1

                request = {
                    'operation': 'eval_template',
                    'setup': setup,
                    'params': tuple(params),
                    'request_id': request_id,
                    'worker_id': worker_id
                }
                self.request_queue.put(request)

                # Poll for response in shared dict
                start_time = time.time()
                while time.time() - start_time < 30:
                    if request_id in self.response_dict:
                        result = self.response_dict.pop(request_id)
                        if 'error' in result:
                            raise RuntimeError(f"GPU server error: {result['error']}")
                        return result
                    time.sleep(0.001)  # 1ms polling interval
                raise TimeoutError('GPU server did not respond in time')

            def eval_chisq(self, spec, templ, poly, espec=None):
                import os
                import time
                # Generate unique request ID
                request_id = f"{os.getpid()}_{self.request_counter}"
                self.request_counter += 1

                request = {
                    'operation': 'eval_chisq',
                    'spec': spec,
                    'templ': templ,
                    'poly': poly,
                    'espec': espec,
                    'request_id': request_id
                }
                self.request_queue.put(request)

                # Poll for response in shared dict
                start_time = time.time()
                while time.time() - start_time < 30:
                    if request_id in self.response_dict:
                        result = self.response_dict.pop(request_id)
                        if 'error' in result:
                            raise RuntimeError(f"GPU server error: {result['error']}")
                        return result
                    time.sleep(0.001)
                raise TimeoutError('GPU server did not respond in time')

            def convolve_vsini(self, lam, templ, vsini):
                import os
                import time
                # Generate unique request ID
                request_id = f"{os.getpid()}_{self.request_counter}"
                self.request_counter += 1

                request = {
                    'operation': 'convolve_vsini',
                    'lam': lam,
                    'templ': templ,
                    'vsini': vsini,
                    'request_id': request_id
                }
                self.request_queue.put(request)

                # Poll for response in shared dict
                start_time = time.time()
                while time.time() - start_time < 30:
                    if request_id in self.response_dict:
                        result = self.response_dict.pop(request_id)
                        if 'error' in result:
                            raise RuntimeError(f"GPU server error: {result['error']}")
                        return result
                    time.sleep(0.001)
                raise TimeoutError('GPU server did not respond in time')

        return GPUServerClient(_gpu_server_queues)

    return None
