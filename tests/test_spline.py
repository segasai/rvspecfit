import rvspecfit.spliner
import numpy as np
import scipy.interpolate


def test_lin():
    x = np.linspace(1000, 2000, 1000)
    y = 0.00001 * x**2 + np.random.normal(size=len(x))
    xnew = np.random.uniform(1000, 2000, size=10000)
    yref = scipy.interpolate.CubicSpline(x, y, bc_type='natural')(xnew)
    ymy = rvspecfit.spliner.Spline(x, y, logstep=False)(xnew)
    assert (np.allclose(yref, ymy))


def test_loglin():
    x = 10**np.linspace(3, 4, 1000)
    y = np.sin(x / 10) + np.random.normal(size=len(x))
    xnew = np.random.uniform(1000, 2000, size=10000)
    yref = scipy.interpolate.CubicSpline(x, y, bc_type='natural')(xnew)
    ymy = rvspecfit.spliner.Spline(x, y, logstep=True)(xnew)
    assert (np.allclose(yref, ymy))
