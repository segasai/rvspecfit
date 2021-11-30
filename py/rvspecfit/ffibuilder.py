# in a separate file "package/foo_build.py"
import cffi
import os

ffibuilder = cffi.FFI()
path = os.path.dirname(os.path.realpath(__file__))
ffibuilder.set_source("rvspecfit._spliner",
                      open(path + '/src/spliner.c', 'r').read())
ffibuilder.cdef("""
void construct(double *xs, double *ys, int N,
double *A, double *B, double *C, double *D, double *h);
int evaler(double *evalx, int nevalx,  int N, double *xs,
           double *hs, double *As, double *Bs, double *Cs,
           double *Ds, int logstep, double *ret);

""")
if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
