"""
Functions for bins
"""

from typing import Union, Tuple, List

import numpy as np


def edge2center(edgebins: np.ndarray) -> np.ndarray:
    """
    turn edge bins to center bins.
    """
    return edgebins[:-1] + 0.5 * (edgebins[1:] - edgebins[:-1])


def get_linbin(
    st: float, ed: float, num: int, offset: int = 1, center: bool = True
) -> np.ndarray:
    """
    Get linear center/edge bins.

    Args:
        st (float): The starting value of the bins.
        ed (float): The ending value of the bins.
        num (int): The number of bins.
        offset (int, optional): The offset value for the bins. Defaults to 1.
        center (bool, optional): Determines if the bins are centered. Defaults to True.

    Returns:
        numpy.ndarray: An array representing the linear bins.
    """
    lbin = np.linspace(st, ed, num + offset)

    if center:
        lbin = edge2center(lbin)

    return lbin


def set_resbins(
    width: float, nx: int, ny: int, offset: int = 1, center: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    set stack result map bins.

    Args:
        width (float): The width of the bins.
        nx (int): The number of bins in the horizontal direction.
        ny (int): The number of bins in the vertical direction.
        offset (int, optional): The offset value for the bins. Defaults to 1.
        center (bool, optional): Determines if the bins are centered. Defaults to True.

    Returns:
        tuple: A tuple containing the horizontal and vertical bins.
    """

    # set horizental and vertical bins
    hbin = get_linbin(-width, width, nx, offset=offset, center=center)
    vbin = get_linbin(-width, width, ny, offset=offset, center=center)

    return (hbin, vbin)


def get_id_edge(
    coordinate: float, bins: Union[list, np.ndarray], right: bool = False
) -> np.int64:
    """
    get coordinate corresponding pixel index in the map.

    Notes:
        we using the numpy.digitize() function to get the index of the bin that the coordinate falls into.
        The right parameter is used to specify whether the coordinate should be included in the rightmost bin.
        The return index of numpy.digitize() with less equal to zero is not valid, large than the size of bins is also invalid.
        And the first index of digitize() is one, so we minus one for zero starting index.
    """

    index = (
        np.digitize(coordinate, bins=bins, right=right) - 1
    )  # minus one is need using digitize()

    return index


def get_ids_edge(
    coordinates: List[float],
    bins: Tuple[List[float], List[float], List[float]],
    right: bool = False,
) -> Tuple[np.int64, np.int64, np.int64]:
    """
    get coordinates corresponding pixel indices in the map.

    Args:
        coordinates (list[float, float, float]): the coordinates of galaxy, given by X, Y, and frequency.
        bins (tuple[list,list,list]): the bins of X, Y, and frequency.
        right (bool): if True, the coordinate should be included in the rightmost bin.

    Returns:
        the indices of X, Y, and frequency.
    """
    x, y, f = coordinates
    xbin, ybin, fbin = bins

    x_id = get_id_edge(x, xbin, right)
    y_id = get_id_edge(y, ybin, right)
    f_id = get_id_edge(f, fbin, right)

    return x_id, y_id, f_id
