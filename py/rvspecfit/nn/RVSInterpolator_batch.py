"""
Batched GPU-enabled neural network interpolator for efficient processing
of multiple spectra simultaneously.
"""

import torch
import numpy as np
import logging
from . import RVSInterpolator


class RVSInterpolatorBatch:
    """
    Wrapper around RVSInterpolator that processes batches of parameters
    efficiently on GPU
    """

    def __init__(self, kwargs, batch_size=128):
        """
        Parameters
        ----------
        kwargs : dict
            Configuration dictionary (same as RVSInterpolator)
        batch_size : int
            Number of spectra to process simultaneously on GPU
        """
        self.interpolator = RVSInterpolator.RVSInterpolator(kwargs)
        self.batch_size = batch_size
        self.device = self.interpolator.device

        logging.info(f'Batched NN interpolator initialized on {self.device} '
                    f'with batch_size={batch_size}')

    def __call__(self, x):
        """
        Evaluate for a single parameter vector (maintains compatibility)

        Parameters
        ----------
        x : array_like
            Parameter vector

        Returns
        -------
        spec : ndarray
            Spectrum
        """
        return self.interpolator(x)

    def batch_eval(self, params_batch):
        """
        Evaluate spectra for a batch of parameters on GPU

        Parameters
        ----------
        params_batch : array_like, shape (n_spectra, n_params)
            Batch of parameter vectors

        Returns
        -------
        specs : ndarray, shape (n_spectra, n_pixels)
            Batch of spectra
        """
        params_batch = np.asarray(params_batch, dtype=np.float32)
        n_spectra = params_batch.shape[0]

        # Process in batches to avoid OOM
        all_specs = []

        with torch.inference_mode():
            for i in range(0, n_spectra, self.batch_size):
                end_idx = min(i + self.batch_size, n_spectra)
                batch = params_batch[i:end_idx]

                # Move to GPU and evaluate
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                specs_batch = self.interpolator.nni(batch_tensor)

                # Move back to CPU and convert
                specs_np = specs_batch.cpu().detach().numpy().astype(np.float64)

                # Apply exponential with clipping
                specs_np = np.exp(np.clip(specs_np, -300, 300))

                all_specs.append(specs_np)

        return np.vstack(all_specs)
