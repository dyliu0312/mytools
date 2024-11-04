"""
estimate the signal level
"""
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def get_int(x):
    """turn the float number to integer, using np.rint"""
    x = np.array(x)
    return np.rint(x).astype(int)

def gaussian_function(x:np.ndarray,
                      amp:Optional[float]=None, 
                      sigma:float=1,
                      c:Optional[float]=None, 
                      mu:float=0):
    """A simple Gaussian function"""
    
    if amp is None:
        amp = 1/np.sqrt(2*np.pi*sigma**2)
    if c is None:
        c = 0
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c

def fit_func(x:np.ndarray, y:np.ndarray, func:Callable, bounds:Optional[Tuple]=None):
    """
    Fit a function to the given data using 'curve fit'.
    
    This function is designed to be used as a helper function for the 'curve_fit' function from 'scipy.optimize'.
    
    It takes the x and y data points, the function to fit, and optional bounds for the parameters.
    
    The function returns the fitted y values, the parameters of the fitted function, and the covariance matrix of the parameters.
    """

    if bounds is not None:
        para, cov, *_ = curve_fit(func, x, y, bounds=bounds)
    else:
        para, cov, *_= curve_fit(func, x, y)

    fit_y = func(x, *para)

    return fit_y, para, cov

def get_start_end_pixel_index(x_min:float=-0.5, 
                              x_max:float=0.5, 
                              shape:int=120, 
                              lim:Tuple[float, float]=(-3, 3),
                              offset:int=1):
    """
    for given xtick range, get the start and end pixel index
    """
    xx = np.linspace(lim[0], lim[1], shape)
    
    x_start = np.argmin(np.abs(xx - x_min))
    x_end = np.argmin(np.abs(xx - x_max)) + offset

    return x_start, x_end

def get_gaussian_fit(x:np.ndarray, 
                     y:np.ndarray, 
                     f:Callable=gaussian_function, 
                     bounds:Optional[Tuple]=([-np.inf, 0,  -np.inf], [np.inf, 0.5,  np.inf]),
                     print_info:bool=True):
    """
    get the Gaussian fit of the given data
    """
    fit_res = fit_func(x, y, f, bounds=bounds)
    
    if print_info:
        print('Gauss fit parameters a, sgima, c (and mu): %s'%fit_res[1])
        print('Gauss fit covariance of a, sgima, c (and mu): %s'%fit_res[2])
        
    return fit_res

def fit_yprofile(data,
                 x_range=[-0.5, 0.5], 
                 xlim=[-3,3], 
                 ylim=[-3,3],
                 show_fit=True,
                 **kwargs):
    """
    fit the y profile of the given data
    
    Args:
        data: the data to fit
        x_range: the x range to be averaged along the x axis
        xlim: the x range of the data
        ylim: the y range of the data
        show_fit: whether to show the fit result
        **kwargs: the kwargs for the 'get_Gaussian_fit' function
    
    Return:
        Tuple[y, yprofile, fit_y, para, cov]: the y axis, the y profile, \
            the fitted y profile, the parameters of the fitted function, \
                and the covariance matrix of the parameters.
    """
    sy, sx = data.shape
    
    # cut the data to the given x range to estimate the width
    x_st, x_ed = get_start_end_pixel_index(*x_range, shape=sx, lim=xlim, offset=1)
    data_cut = data[:,x_st:x_ed]
    
    # fit the y_profile to estimate the width
    y_profile = np.mean(data_cut, axis=1)
    yy = np.linspace(*ylim, sy) # type: ignore
    fit_y, para, cov = get_gaussian_fit(yy, y_profile, **kwargs)
    
    if show_fit:
        plt.plot(y_profile, label='y profile')
        plt.plot(fit_y, label='Gaussian fit')
        plt.legend()

    return yy, y_profile, fit_y, para, cov
    
def get_center_outer_background(data,
                                width, 
                                x_range=[-0.5, 0.5], 
                                xlim=[-3,3], 
                                ylim=[-3,3], 
                                shift_unit=2):
    """
    get the center, outer and background region.
    
    Args:
        data (np.ndarray): the data to be fitted
        x_range (List[float]): the x range to be fitted
        xlim (List[float]): the x limit of the data
        ylim (List[float]): the y limit of the data
        shift_unit (int): the shift unit of the outer region
        **kwargs (dict): the parameters for the Gaussian fit
    Returns:
        Tuple[np.ndarray]: the center, the outer left, the outer right , and the background.
        
    
    """
    sy, sx = data.shape
    npix_unit_x = sx / (xlim[1] - xlim[0])
    npix_unit_y = sy / (ylim[1] - ylim[0])
    width_pix = get_int(width * npix_unit_y)
    print('width: %s'%width)
    print('width_pix: %s'%width_pix)
    
    # cut the data to the given x range to estimate the width
    x_st, x_ed = get_start_end_pixel_index(*x_range, shape=sx, lim=xlim, offset=1)
    data_cut = data[:,x_st:x_ed]
    
    # get the center, outer and background region
    ## the center area use the gauss 1 sigma range
    y_st = sy // 2 - width_pix
    y_ed = sy // 2 + width_pix
    # y_st, y_ed = get_start_end_pixel_index(-width, width, shape=data.shape[0], lim=ylim, offset=1)
    center =  data_cut[y_st:y_ed, :]
    
    ## the background area use the 3-4 sigma range with a x width same with center
    bk_top_ids = y_st + width_pix*4, y_ed + width_pix*3
    bk_bottom_ids = y_st - width_pix*3, y_ed - width_pix*4
    # print(bk_top_ids, bk_bottom_ids)
    # bk_top_ids = get_start_end_pixel_index(width*3, width*4, shape=data.shape[0], lim=ylim, offset=1)
    # bk_bottom_ids = get_start_end_pixel_index(-width*4, -width*3, shape=data.shape[0], lim=ylim, offset=1)
    bk_list = list(range(bk_top_ids[0], bk_top_ids[1])) + list(range(bk_bottom_ids[0], bk_bottom_ids[1]))
    background = data[bk_list, x_st:x_ed]

    ## the outer area use the the center region shift along x-axis
    left_ids = get_int([x_st - shift_unit*npix_unit_x, x_ed - shift_unit*npix_unit_x])
    right_ids = get_int([x_st + shift_unit*npix_unit_x, x_ed + shift_unit*npix_unit_x])
    # print(left_ids, right_ids)
    # left_ids = get_start_end_pixel_index(x_range[0]-shift_unit, x_range[1]-shift_unit, shape=data.shape[1], lim=xlim, offset=0)
    # right_ids = get_start_end_pixel_index(x_range[0]+shift_unit, x_range[1]+shift_unit, shape=data.shape[1], lim=xlim, offset=0)
    left = data[y_st:y_ed, left_ids[0]:left_ids[1]]
    right = data[y_st:y_ed, right_ids[0]:right_ids[1]]
    
    return center, left, right, background

