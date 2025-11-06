"""
Functions to fit and subtract halo contributions.

This script is an optimized version of mytools/halo.py for more robust and flexible halo fitting.
"""

from typing import Callable, Generator, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter1d, gaussian_filter, uniform_filter
from mytools.bins import get_id_edge


# --- 1. Coordinates and Masking Utilities ---
def get_coord(
    n: int = 120,
    xlim: Sequence[float] = (-3, 3),
    ylim: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate coordinate grid for 2D modeling.

    Parameters
    ----------
    n : int, optional
        Number of grid points in each dimension, by default 120
    xlim : Sequence[float], optional
        x-coordinate limits, by default (-3, 3)
    ylim : Sequence[float], optional
        y-coordinate limits, by default None (same as xlim)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x and y coordinate arrays
    """
    if ylim is None:
        ylim = xlim
    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(x, y)
    return X, Y


def get_r_theta(
    center: Sequence[float] = [0, 0],
    coord: Optional[Sequence[np.ndarray]] = None,
    get_theta: bool = False,
) -> Union[np.ndarray, Sequence[np.ndarray]]:
    """
    Compute radial (and angular) coordinates from 2D meshgrid.
    """
    if coord is None:
        coord = get_coord()
    x, y = coord
    x = x - center[0]
    y = y - center[1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if get_theta:
        return r, theta
    else:
        return r


def get_mask_square(
    coord: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    xlim: Tuple[float, float] = (-1, 1),
    ylim: Tuple[float, float] = (-1, 1),
) -> np.ndarray:
    """
    Create a square mask for region selection
    """
    if coord is None:
        coord = get_coord()
    x, y = coord
    x1, x2 = xlim
    y1, y2 = ylim
    mask = (x1 < x) & (x < x2) & (y1 < y) & (y < y2)
    return mask


def _get_mask_sector1(
    r: np.ndarray,
    theta: np.ndarray,
    rlim: Sequence[float] = (0.5, 1),
    thetalim: Sequence[float] = (-np.pi / 4, np.pi / 4),
) -> np.ndarray:
    """
    Create a circular sector mask (internal utility)
    """
    mask = (rlim[0] < r) & (r < rlim[1]) & (thetalim[0] < theta) & (theta < thetalim[1])
    return mask


def _get_mask_sector2(
    r: np.ndarray,
    theta: np.ndarray,
    rlim: Sequence[float] = (0.5, 1),
    thetalim: Sequence[float] = (-np.pi / 4, np.pi / 4),
) -> np.ndarray:
    """
    Create a two-sector mask with offset center (internal utility)
    """

    # Create two symmetric sectors rotated by ±90 degrees
    mask1 = _get_mask_sector1(
        r,
        theta,
        rlim,
        (thetalim[0] - np.pi / 2, thetalim[1] - np.pi / 2),
    )
    mask2 = _get_mask_sector1(
        r,
        theta,
        rlim,
        (thetalim[0] + np.pi / 2, thetalim[1] + np.pi / 2),
    )
    mask = mask1 | mask2
    return mask


def _get_mask_sector4(
    r1: np.ndarray,
    theta1: np.ndarray,
    r2: np.ndarray,
    theta2: np.ndarray,
    rlim: Sequence[float] = (0.5, 1),
    thetalim: Sequence[float] = (-np.pi / 4, np.pi / 4),
) -> np.ndarray:
    """
    Create a four-sector mask with two offset centers (internal utility)
    """
    mask1 = _get_mask_sector2(r1, theta1, rlim, thetalim)
    mask2 = _get_mask_sector2(r2, theta2, rlim, thetalim)
    return mask1 | mask2


def get_mask_sector(
    rlim: Sequence[float] = (0.2, 5),
    thetalim: Sequence[float] = (-np.pi / 4, np.pi / 4),
    rc: Sequence[float] = (1, 0),
    lc: Sequence[float] = (-1, 0),
    coord: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """
    Create a four-sector mask with two offset centers.

    Args:
        rlim (Sequence[float]): The radial limits of the mask.
        thetalim (Sequence[float]): The angular limits of the mask.
        rc (Sequence[float]): The center of the first sector.
        lc (Sequence[float]): The center of the second sector.
        coord (Sequence[np.ndarray]): The coordinate grid.
    Returns:
        np.ndarray: The mask array.
    """
    if coord is None:
        coord = get_coord()

    r1, theta1 = get_r_theta(rc, coord, get_theta=True)
    r2, theta2 = get_r_theta(lc, coord, get_theta=True)

    return _get_mask_sector4(r1, theta1, r2, theta2, rlim, thetalim)


# --- Weighting functions ---


def generate_local_noise_weights(
    data: np.ndarray,
    smoothing_scale: float = 5.0,
    local_variance_scale: Union[int, Sequence[int]] = 3,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Generates weights based on the local variance of the data.

    This method assumes that "noise" is the part of the data that remains
    after subtracting a smoothed version of it. The weights are calculated as
    the inverse of the local variance of this noise.

    This is useful for down-weighting regions that are "noisy" or "wiggly"
    without being tied to a specific coordinate system like radial bins.

    Args:
        data (np.ndarray): The 2D data map.
        smoothing_scale (float): The sigma of the Gaussian filter used to
                                 separate signal from noise.
        local_variance_scale (int or Sequence[int]): The size of the boxcar filter used to
                                      calculate the local variance of the noise.
        epsilon (float): A small number to add to the variance to avoid
                         division by zero.

    Returns:
        np.ndarray: A weight map with the same shape as the data.
    """
    # 1. Estimate signal by smoothing the data
    signal_estimate = gaussian_filter(data, sigma=smoothing_scale)

    # 2. Estimate noise as the residual
    noise_estimate = data - signal_estimate

    # 3. Calculate local variance of the noise
    # Use a uniform filter (boxcar) on the squared noise to get local variance
    local_variance = uniform_filter(noise_estimate**2, size=local_variance_scale)  # pyright: ignore[reportArgumentType]

    # 4. Calculate weights as inverse variance
    weights = 1.0 / (local_variance + epsilon)

    # 5. Normalize weights to have a mean of 1, which is good practice
    weights /= np.mean(weights)

    return weights


# --- Smoothing function ---


def smooth_profile(profile: np.ndarray, window_size: int) -> np.ndarray:
    """
    Smooth a 1D profile using a uniform filter (moving average).

    Args:
        profile (np.ndarray): The 1D profile to smooth.
        window_size (int): The size of the smoothing window.

    Returns:
        np.ndarray: The smoothed profile.
    """
    if window_size <= 1:
        return profile
    return uniform_filter1d(profile, size=window_size)


# --- Core fitting functions ---
def yield_mask_data(
    *args: np.ndarray,
    mask: Optional[np.ndarray] = None,
    flat: bool = False,
) -> Generator[np.ndarray]:
    """
    provides a generator that yields masked (and flattened) data arrays.

    Args:
        args (np.ndarray): The data arrays to be masked and flattened.
        mask (np.ndarray, optional): The mask to apply to the data. Defaults to None.
        flat (bool, optional): Whether to flatten the data. Defaults to False.
    Yields:
        np.ndarray: The masked and flattened data arrays.
    """
    for arg in args:
        if mask is not None:
            if flat:
                yield arg[mask].flatten()
            else:
                yield arg[mask]
        else:
            if flat:
                yield arg.flatten()
            else:
                yield arg


def info_fitness(
    data: np.ndarray, fit_data: np.ndarray, return_res: bool = True
) -> Union[np.ndarray, None]:
    """
    print the goodness of fit for the given data and fit data
    """
    # Calculate goodness of fit
    residuals = data - fit_data
    chi2 = np.sum(residuals**2)
    print("\nGoodness of fit:")
    print(f"Chi-squared: {chi2:.3f}")
    print(f"RMS residual: {np.sqrt(np.mean(residuals**2)):.3f}")

    if return_res:
        return residuals


def halo_fit(
    data: np.ndarray,
    rbin_e: np.ndarray = np.arange(0, 6, 0.04),
    lc: list = [-1, 0],
    rc: list = [1, 0],
    smooth_window: int = 0,
    weight: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    coord: Optional[Sequence[np.ndarray]] = None,
    info_fit: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced halo fitting function with weighted fitting and profile smoothing.

    This function retains the core logic of the original `halo_fit` but adds
    support for weighted least squares and allows for smoothing the resulting
    halo profile to achieve a smoother 2D fit.

    Args:
        data (np.ndarray): 2D data to be fitted.
        rbin_e (np.ndarray): The edge of radius bins.
        lc (list): The center of the left halo.
        rc (list): The center of the right halo.
        smooth_window (int): The size of the window for profile smoothing. \
            If 0 or 1, no smoothing is applied. Defaults to 0.
        weight (np.ndarray, optional): Weights for each data point. If None,
                                        equal weights are used (flat weighting).
                                        Defaults to None.
        mask (np.ndarray, optional): Mask to apply to the data. Defaults to None.
        coord (Sequence[np.ndarray], optional): Coordinates of the data. Defaults to None. \
            If None, the function will attempt to infer the coordinates from the data shape.
        info_fit (bool, optional): If True, print the fitting result. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - fit_map: The 2D map of the fitted halo model.
            - fit_paras: The fitted 1D halo profile (smoothed if requested).
    """
    if coord is None:
        coord = get_coord(data.shape[0])
    if weight is None:
        weight = np.ones_like(data)

    r1 = get_r_theta(lc, coord)
    r2 = get_r_theta(rc, coord)

    r1_ids = get_id_edge(r1, rbin_e)  # pyright: ignore[reportArgumentType]
    r2_ids = get_id_edge(r2, rbin_e)  # pyright: ignore[reportArgumentType]

    m_data, m_weight, m_r1_ids, m_r2_ids = yield_mask_data(
        data,
        weight,
        r1_ids,
        r2_ids,
        mask=mask,
        flat=True,  # pyright: ignore[reportArgumentType]
    )

    def get_fitmap(profile, r1_ids, r2_ids):
        padded_profile = np.pad(
            profile,
            (0, max(0, max(r1_ids.max(), r2_ids.max()) - len(profile) + 1)),
            "edge",
        )
        return padded_profile[r1_ids] + padded_profile[r2_ids]

    def error(paras, data, r1_ids, r2_ids, weight):
        fitmap = get_fitmap(paras, r1_ids, r2_ids)
        residual = data - fitmap
        return np.sqrt(weight) * residual

    num_bins = len(rbin_e) - 1
    init_paras = np.random.random(size=num_bins) * np.nanmax(m_data)

    result = least_squares(
        error, init_paras, args=(m_data, m_r1_ids, m_r2_ids, m_weight)
    )
    fit_paras = result.x

    if info_fit:
        # print(result)
        fit_map = get_fitmap(fit_paras, m_r1_ids, m_r2_ids)
        info_fitness(m_data, fit_map)

    if smooth_window > 1:
        fit_paras = smooth_profile(fit_paras, smooth_window)

    fit_map = get_fitmap(fit_paras, r1_ids, r2_ids)

    return fit_map, fit_paras


def halo_subtract(
    data: Union[np.ndarray, Sequence[np.ndarray]],
    func_halo_fit: Callable[..., Tuple[np.ndarray, np.ndarray]] = halo_fit,
    rbin_e: np.ndarray = np.arange(0, 6, 0.04),
    lc: list = [-1, 0],
    rc: list = [1, 0],
    smooth_window: int = 0,
    weight: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    coord: Optional[Sequence[np.ndarray]] = None,
    info_fit: bool = True,
):
    """
    for each data, fit a halo profile, and subtract it from the data.
    Args:
        data (list or np.ndarray): the data to be fitted and subtracted.
        func_halo_fit (Callable): the function to fit the halo profile.
        rbin_e (np.ndarray): the edge of the radius bins.
        lim (list[float,float]): the x range of the data.
        costheta (float): the valid area are selected within -costheta to costheta.
        print_fit_res (bool): whether to print the result of the fitting.
    Return:
        fitted_map_list, residuals_list, mask_bool_array
    """
    is_single = isinstance(data, np.ndarray)
    if is_single:
        data = [data]

    fit = []
    res = []
    fit_mask = None

    for d in data:
        (
            fitmap,
            _,
        ) = func_halo_fit(
            d, rbin_e, lc, rc, smooth_window, weight, mask, coord, info_fit
        )
        fit.append(fitmap)
        res.append(d - fitmap)

    if is_single:
        return fit[0], res[0], fit_mask
    else:
        return fit, res, fit_mask
