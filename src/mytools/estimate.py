"""
estimate the signal level
"""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip

from mytools.plot import plot_line_diff


def get_int(x):
    """turn the float number to integer, using np.rint"""
    x = np.array(x)
    return np.rint(x).astype(int)


def get_start_end_pixel_index(
    x_min: float = -0.5,
    x_max: float = 0.5,
    shape: int = 120,
    lim: Sequence[float] = (-3, 3),
    offset: int = 1,
):
    """
    for given xtick range, get the start and end pixel index
    """
    xx = np.linspace(lim[0], lim[1], shape)

    x_start = np.argmin(np.abs(xx - x_min))
    x_end = np.argmin(np.abs(xx - x_max)) + offset

    return x_start, x_end


def get_gaussian_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    sigma_clip_sigma: float = 3.0,
    bounds: Optional[dict] = None,
    print_info: bool = True,
    **kwargs,
):
    """
    Get the Gaussian fit of the given data using astropy.modelling with sigma clipping.

    Args:
        x (np.ndarray): The x data.
        y (np.ndarray): The y data.
        weights (Optional[np.ndarray], optional): The weights for the fitting. Defaults to None.
        sigma_clip_sigma (float, optional): The sigma for sigma clipping. Defaults to 3.0.
        bounds (Optional[dict], optional): The bounds for the parameters. Defaults to None.
        print_info (bool, optional): Whether to print the fit information. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, models.CompoundModel]:
            The fitted y values, the parameters of the fitted function,
            the covariance matrix of the parameters, and the fitted model.
    """
    # Sigma clipping on the data
    clipped_data: np.ma.MaskedArray = sigma_clip(y, sigma=sigma_clip_sigma, masked=True)  # pyright: ignore[reportAssignmentType]
    x_clipped = x[~clipped_data.mask]
    y_clipped = clipped_data.data[~clipped_data.mask]
    weights_clipped = None
    if weights is not None:
        weights_clipped = weights[~clipped_data.mask]

    if len(x_clipped) < 3:  # Not enough points to fit for 3 parameters
        print("Warning: Not enough data points to fit after sigma clipping.")
        return np.zeros_like(x), np.array([0, 0, 0]), None, None

    # Initial guess for the parameters
    amp_init = np.max(y_clipped) - np.median(y_clipped)
    stddev_init = (x_clipped.max() - x_clipped.min()) / 8.0  # A reasonable guess
    c_init = np.median(y_clipped)

    # Define the model: Gaussian + constant, with fixed mean=0
    model_init = models.Gaussian1D(  # pyright: ignore[reportOperatorIssue]
        amplitude=amp_init, mean=0, stddev=stddev_init
    ) + models.Const1D(amplitude=c_init)
    model_init.mean_0.fixed = True

    # Apply bounds if provided
    if bounds:
        for param_name, bound_tuple in bounds.items():
            param = getattr(model_init, param_name)
            param.bounds = bound_tuple

    # Fitter
    fitter = fitting.LevMarLSQFitter()

    # Fit the model to the clipped data
    fitted_model = fitter(
        model_init, x_clipped, y_clipped, weights=weights_clipped, **kwargs
    )

    # Get results
    fit_y = fitted_model(x)
    para = fitted_model.parameters
    cov = None
    if fitter.fit_info.get("param_cov") is not None:
        cov = fitter.fit_info["param_cov"]

    if print_info:
        print("Parameter names:")
        print(fitted_model.param_names)
        print("Gauss fit parameters: %s" % list(para))
        if cov is not None:
            print("Gauss fit covariance: \n%s" % cov)
        print(f"{np.sum(clipped_data.mask)} points clipped during sigma clipping.")

    return fit_y, para, cov, fitted_model


def fit_yprofile(
    data: np.ndarray,
    x_range: Sequence[float] = [-0.5, 0.5],
    xlim: Sequence[float] = [-3, 3],
    ylim: Sequence[float] = [-3, 3],
    show_fit: bool = True,
    weights: Optional[np.ndarray] = None,
    **kwargs,
):
    """
    fit the y profile of the given data

    Args:
        data: the data to fit
        x_range: the x range to be averaged along the x axis
        xlim: the x range of the data
        ylim: the y range of the data
        show_fit: whether to show the fit result
        weights: the weights for the data
        **kwargs: the kwargs for the 'get_Gaussian_fit' function

    Return:
        Tuple[y, yprofile, fit_y, para, cov]: the y axis, the y profile, \
            the fitted y profile, the parameters of the fitted function, \
                and the covariance matrix of the parameters.
    """
    sy, sx = data.shape

    # cut the data to the given x range to estimate the width
    x_st, x_ed = get_start_end_pixel_index(*x_range, shape=sx, lim=xlim, offset=1)
    data_cut = data[:, x_st:x_ed]

    # fit the y_profile to estimate the width
    y_profile = np.mean(data_cut, axis=1)
    yy = np.linspace(*ylim, num=sy)

    # get weights for y_profile fitting
    fit_weights = None
    if weights is not None:
        if weights.shape != data.shape:
            raise ValueError(
                f"Shape of weights {weights.shape} must match shape of data {data.shape}."
            )
        weights_cut = weights[:, x_st:x_ed]
        fit_weights = np.mean(weights_cut, axis=1)

    # Default bounds for astropy fitting
    bounds = {
        "amplitude_0": (-np.inf, np.inf),  # amp > 0
        "stddev_0": (0, 0.5),  # sigma
    }

    # Update with any user-provided bounds
    if "bounds" in kwargs:
        bounds.update(kwargs["bounds"])

    kwargs["bounds"] = bounds

    fit_y, para, cov, _ = get_gaussian_fit(yy, y_profile, weights=fit_weights, **kwargs)

    if show_fit:
        _ = plot_line_diff(
            yy,
            y_profile,
            fit_y,
            labels=["Y-profile", "Gaussian fit", "residual"],
            xlabel="Y",
            ylabel=r"T [$\mu$K]",
        )

    return yy, y_profile, fit_y, para, cov


def get_center_outer_background(
    data, width, x_range=[-0.5, 0.5], xlim=[-3, 3], ylim=[-3, 3], shift_unit=2
):
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
    print("width: %s" % width)
    print("width_pix: %s" % width_pix)

    # cut the data to the given x range to estimate the width
    x_st, x_ed = get_start_end_pixel_index(*x_range, shape=sx, lim=xlim, offset=1)
    data_cut = data[:, x_st:x_ed]

    # get the center, outer and background region
    ## the center area use the gauss 1 sigma range
    y_st = sy // 2 - width_pix
    y_ed = sy // 2 + width_pix
    # y_st, y_ed = get_start_end_pixel_index(-width, width, shape=data.shape[0], lim=ylim, offset=1)
    center = data_cut[y_st:y_ed, :]

    ## the background area use the 3-4 sigma range with a x width same with center
    bk_top_ids = y_st + width_pix * 4, y_ed + width_pix * 3
    bk_bottom_ids = y_st - width_pix * 3, y_ed - width_pix * 4
    # print(bk_top_ids, bk_bottom_ids)
    # bk_top_ids = get_start_end_pixel_index(width*3, width*4, shape=data.shape[0], lim=ylim, offset=1)
    # bk_bottom_ids = get_start_end_pixel_index(-width*4, -width*3, shape=data.shape[0], lim=ylim, offset=1)
    bk_list = list(range(bk_top_ids[0], bk_top_ids[1])) + list(
        range(bk_bottom_ids[0], bk_bottom_ids[1])
    )
    background = data[bk_list, x_st:x_ed]

    ## the outer area use the the center region shift along x-axis
    left_ids = get_int(
        [x_st - shift_unit * npix_unit_x, x_ed - shift_unit * npix_unit_x]
    )
    right_ids = get_int(
        [x_st + shift_unit * npix_unit_x, x_ed + shift_unit * npix_unit_x]
    )
    # print(left_ids, right_ids)
    # left_ids = get_start_end_pixel_index(x_range[0]-shift_unit, x_range[1]-shift_unit, shape=data.shape[1], lim=xlim, offset=0)
    # right_ids = get_start_end_pixel_index(x_range[0]+shift_unit, x_range[1]+shift_unit, shape=data.shape[1], lim=xlim, offset=0)
    left = data[y_st:y_ed, left_ids[0] : left_ids[1]]
    right = data[y_st:y_ed, right_ids[0] : right_ids[1]]

    return center, left, right, background


def get_signal_level(
    data,
    weights: Optional[np.ndarray] = None,
    x_range=[-0.5, 0.5],
    x_num=20,
    xlim=[-3, 3],
    ylim=[-3, 3],
    shift_unit=2,
    show_fit=True,
    **kwargs,
):
    """
    get the signal level of the given data

    Args:
        data (np.ndarray): the data to be fitted
        weights (Optional[np.ndarray], optional): The weights for the fitting. Defaults to None.
        x_range (List[float]): the x range to be fitted
        x_num (int): the number points of x axis
        xlim (List[float]): the x limit of the data
        ylim (List[float]): the y limit of the data
        shift_unit (int): the shift unit of the outer region
        show_fit (bool): whether to plot the fit result
        **kwargs (dict): the parameters for the `get_Gaussian fit`
    Returns:
        Tuple[List]:
        1. the x line point list for `plot_profiles`,
        2. the y line point list for `plot_profiles`,
        3. the extracted and flatten center, left, right, and background area data points for `plot_hist_result`,
        4. the parameters of the fitted yprofile function,
        5. the covariance matrix of the parameters.
    """

    # fit the y profile
    yy, yprofile, yfit, paras, cov = fit_yprofile(
        data, x_range, xlim, ylim, show_fit, weights=weights, **kwargs
    )

    # get the center, left, right and background
    # With mean fixed, paras are [amplitude_0, x_mean_0, stddev_0, amplitude_1]
    # width is stddev_0, which is paras[2]
    width = paras[2] if paras is not None else 0.1
    clrb = get_center_outer_background(data, width, x_range, xlim, ylim, shift_unit)

    # get the filament signal level and background level
    tf = np.mean(clrb[0], axis=0)
    tbg = np.mean(clrb[-1], axis=0)

    # get the x axis
    xx = np.linspace(*x_range, x_num)  # pyright: ignore[reportCallIssue]

    # create the line data
    line_x = [yy, yy, xx, xx]
    line_y = [yprofile, yfit, tf, tbg]

    # get flattened data
    clrb_flat = [x.flatten() for x in clrb]

    return line_x, line_y, clrb_flat, paras, cov
