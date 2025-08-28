/* Construct the coefficients of the cubic natural spline 
given the array of x,y positions (xs,ys) 
The spline will then be A[i]*(x-x_i)^3 + B[i]*(x_{i+1}-x)^3+
C[i] * (x-x[i]) + D[i])*(x[i+1]-x)
 */ 

void construct(double *xs, double *ys, int N, double *A, double *B, double *C,
               double *D, double *h) {
  const int N1 = N - 1;
  const int N2 = N - 2;
  const int N3 = N - 3;
  double vs[N2];
  double bs[N1];
  double us[N1];
  double zs[N];
  double cc_dash[N3];
  double dd_dash[N2];
  double hinv[N1];
  const double one_sixth = 1./6;
  // see here https://github.com/segasai/stan-splines
  for (int i = 0; i < N1; i++) {
    h[i] = xs[i + 1] - xs[i];
    hinv[i] = 1. / h[i]; // using inverse reduces the number of divisions
    bs[i] = (ys[i + 1] - ys[i]) * hinv[i];
  }
  for (int i = 0; i < N2; i++) {
    vs[i] = 2 * (h[i + 1] + h[i]);
    us[i] = 6 * (bs[i + 1] - bs[i]);
  }
  const double *cc = h + 1; // offdiagonal
  const double *bb = vs;    // diagonal
  const double *dd = us;    // RHS
  cc_dash[0] = cc[0] / bb[0];
  // thomas algo where we assume symmetric matrix
  // https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
  for (int i = 1; i < N3; i++) {
    cc_dash[i] = cc[i] / (bb[i] - cc[i - 1] * cc_dash[i - 1]);
  }
  dd_dash[0] = dd[0] / bb[0];
  for (int i = 1; i < N2; i++) {
    dd_dash[i] = (dd[i] - cc[i - 1] * dd_dash[i - 1]) /
                 (bb[i] - cc[i - 1] * cc_dash[i - 1]);
  }
  zs[N1] = 0;
  zs[0] = 0;
  zs[N2] = dd_dash[N3];
  // shifted by 1 because we inserted 0 in the beginning and end
  for (int i = N3; i >= 1; i--) {
    zs[i] = dd_dash[i - 1] - cc_dash[i - 1] * zs[i + 1];
  }

  for (int i = 0; i < N1; i++) {
    const double tmp1 = hinv[i] * one_sixth;
    const double tmp2 = h[i] * one_sixth;
    A[i] = zs[i + 1] * tmp1;
    B[i] = zs[i] * tmp1;
    C[i] = ys[i + 1] * hinv[i] - zs[i + 1] * tmp2;
    D[i] = ys[i] * hinv[i] - zs[i] * tmp2;
  }
}

/* 
   evaluate the cubic spline at locations evalx
   given the location of knots xs 
   spacings hs
   and coefficients As, Bs, Cs, Ds
   if logstep==1 we assume that the knots are uniformly spaced in log
   if it is zero they are *linearly* spaced
   The result is written in ret
 */
int evaler(double *evalx, int nevalx, int N, double *xs, double *hs, double *As,
           double *Bs, double *Cs, double *Ds, int logstep, double *ret) {
  int pos[nevalx];
  double x0 = xs[0];
  double xlast = xs[N - 1];
  // some checking that the first and last values are within
  // knot boundary
  if ((evalx[0] < x0) || (evalx[nevalx - 1] < x0)) {
    return -1;
  }
  if ((evalx[0] > xlast) || (evalx[nevalx - 1] > xlast)) {
    return -1;
  }
  if (logstep) {
    double logstep = log(xs[1] / x0);
    double logstep2 = log(xs[2] / xs[1]);
    if (fabs(logstep-logstep2)>1e-10) {return -2;} // validation
    double logx0 = log(x0);
    for (int i = 0; i < nevalx; i++) {
      pos[i] = (int)((log(evalx[i]) - logx0) / logstep);
    }
  } else {
    double step = xs[1] - x0;
    double step2 = xs[2]-xs[1];
    if (fabs(step-step2)>1e-10) {return -2;} // validation
    for (int i = 0; i < nevalx; i++) {
      pos[i] = (int)((evalx[i] - x0) / step);
    }
  }
  for (int i = 0; i < nevalx; i++) {
    int curposl = pos[i];
    double dxl = evalx[i] - xs[curposl];
    double dxr = xs[curposl + 1] - evalx[i];
    ret[i] = As[curposl] * dxl * dxl * dxl + Bs[curposl] * dxr * dxr * dxr +
             Cs[curposl] * dxl + Ds[curposl] * dxr;
  }
  return 0;
}
