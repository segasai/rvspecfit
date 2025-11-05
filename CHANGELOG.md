Changes

* 25.10 Update docs
* 25.10 Switch from fork multiprocessing to spawn
* 25.06 Fix the bookeeping bug, when the DESI spectra- files are fitted. If the object was selected to be fitted based on it's targeting bits, but it does not satistfy (SPECTYPE='STAR' |3e5*Z|<1500) criteria then previously the RR_Z, RR_SPECTYPE would be null
* 25.04 Update the penalty at the edge. Previously it was very sharp, because it was jumping from zero to some value. Now it is smooth, as computed throug distance to the edge of the convex hull
* 25.04 DESI specific. Now if the hessian inversions failed, there will be a warning flag
* 25.04 DESI specific. Now the redrock subtype is also saved in the rvtab
* 25.04 When using BFGS optimizer, I provide initial Inverse hessian, which should help convergence
* 25.01.17 Changed a way logarithmic step is setup. it is now set relative to the center of the wavelength
* 0.6.0.241122 The new MPI mode was added
* 0.6.0 Now the preprocessed information about the templates is stored in the HDF5. That means that all this information has to be regenerated. If you don't want to do that, you should use the 0.5.0 version
