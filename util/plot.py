import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


def get_smooth_curve(x, y):
    x_new = np.linspace(min(x), max(x), 1000)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_new)

    return x_new, y_smooth
