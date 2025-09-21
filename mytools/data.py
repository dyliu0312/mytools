"""
tools for data process.

"""

import os
from typing import List, Union, Callable

import h5py as h5
import numpy as np


# --- os ---
def get_file_dir(path: str) -> str:
    """
    get the file parrent dir
    """
    return os.path.dirname(path)


def get_filename(path: str) -> str:
    """
    get the file name
    """
    file_name = os.path.splitext(os.path.basename(path))[0]
    return file_name


def delete_files(file_paths: List[str]) -> None:
    """
    Deletes files from the given list of file paths.

    Args:
        file_paths (List[str]): A list of file paths to be deleted.

    Prints:
        Messages indicating whether each file was deleted, not found, or caused an error.
    """
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except PermissionError:
            print(f"Permission error: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def is_exist(file_path: str) -> bool:
    """
    check file exists.
    """
    return os.path.exists(file_path)


# --- hdf5 file ---
def visititems_h5(file_path: str, func: Callable = print):
    """
    Recursively visit the group and dataset information in an HDF5 file using the visititems() method.

    Args:
        file_path (str): Path to the HDF5 file.
        func (callable): Function to be applied to each item (group or dataset) in the HDF5 file.
                        Defaults to `print` which will print the name and item type.
    """
    try:
        # Open the HDF5 file in read-only mode
        with h5.File(file_path, "r") as f:
            # Visit all items in the file (groups and datasets) and apply `func` to each
            f.visititems(func)
    except Exception as e:
        print(f"Error accessing HDF5 file: {e}")


def del_data(file_path: str, key: str):
    """
    delete dataset and it's attributes from h5 file.
    """
    with h5.File(file_path, "r+") as f:
        # del attributes
        for k in f[key].attrs.keys():
            del f[key].attrs[k]
        # del data
        del f[key]


def print_dataset_attrs(dataset):
    """
    Prints the attributes of a given dataset if any exist.
    """
    attrs = dataset.attrs
    if attrs:
        print(f"Attributes for '{dataset.name}':")
        for attr_name, attr_value in attrs.items():
            print(f"  {attr_name}: {attr_value}")


def read_h5(
    file_path: str, key: Union[str, List[str]], print_attrs: bool = False
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Read dataset from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        key (str or list): Key(s) to read from the HDF5 file.
            It can be a single key (str) or a list of keys.
        print_attrs (bool): If True, prints the attributes of the datasets
            when reading them.

    Returns:
        out (ndarray or list): The dataset(s) read from the HDF5 file.
    """
    # Ensure the key is either a string or a list of strings
    if not isinstance(key, (str, list)):
        raise TypeError("Key must be a string or a list of strings.")

    # Open the HDF5 file in read-only mode
    with h5.File(file_path, "r") as f:
        available_keys = list(f.keys())

        # Helper function to handle reading and printing attributes for each key
        def get_data_for_key(k):
            if k not in f:
                raise ValueError(
                    f"Key '{k}' not found. Available keys are: {available_keys}"
                )
            obj = f[k]
            if k in f and not obj.shape == (): # type: ignore
                dataset = obj[:] # type: ignore
            else:
                dataset = obj[()] # type: ignore
            if print_attrs:
                print_dataset_attrs(obj)
            return dataset

        # If a single key is provided, return the corresponding data
        if isinstance(key, str):
            return get_data_for_key(key)  # type: ignore

        # If a list of keys is provided, return a list of corresponding data
        elif isinstance(key, list):
            data_list = [get_data_for_key(k) for k in key]
            return data_list  # type: ignore


def save_h5(
    output: str, keys: List[str], data: list, group_name: Union[str, None] = None
):
    """
    Save data to an HDF5 file and create group if group_name is provided.

    **For single dataset, keys and datas should be provided list with one element.
    e.g. keys = ['data',], data = [data,]**

    Args:
        output (str): The path to the output HDF5 file.
        keys (list): A list of keys for the datasets.
        data (list): A list of data arrays to be saved.
        group_name (str, optional): A group paths to be created. Defaults to None.

    """
    with h5.File(output, "a") as f:
        if group_name is not None:
            f.create_group(group_name)

        for k, d in zip(keys, data):
            if group_name is not None:
                dataset_path = f"{group_name}/{k}"
            else:
                dataset_path = k
            f.create_dataset(dataset_path, data=d)


def save_h5_attributes(filepath, file_attributes=None, dataset_attrs=None):
    """
    Save specified attributes in an HDF5 file, with an option to add dataset-level attributes.

    Parameters:
    filepath (str): Path to the HDF5 file. It will create the file if it doesn't exist.
    attributes (dict, optional): A dictionary containing file-level key-value pairs to store as attributes.
    dataset_attrs (dict, optional): A dictionary containing keys as dataset names and values as dictionaries of dataset-level attributes.
                                     For example: {'dataset1': {'units': 'm', 'description': 'Height data'}}.
    """

    if file_attributes is None and dataset_attrs is None:
        raise ValueError("Please provided a kind of attributes at least.")

    with h5.File(filepath, "a") as f:
        if file_attributes is not None:
            # Store file-level attributes in the root of the HDF5 file
            for key, value in file_attributes.items():  # type: ignore
                try:
                    f.attrs[key] = value
                except Exception as e:
                    print(f"Error storing attributes: {e}")
            print(f"Successfully stored file-level attributes in {filepath}")

        if dataset_attrs is not None:
            # Store dataset-specific attributes (if provided)
            for dataset_name, attrs in dataset_attrs.items():
                if dataset_name not in f:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found in {filepath}"
                    )
                dataset = f[dataset_name]
                for attr_name, attr_value in attrs.items():
                    dataset.attrs[attr_name] = attr_value
                print(
                    f"Successfully stored dataset-level attributes for '{dataset_name}'"
                )


# --- generator ---
def split_data_generator(splitsize: int, *data: np.ndarray):
    """
    Splits multiple datasets into chunks of size `splitsize`, ensuring that all datasets have the same
    first dimension size. The generator yields corresponding slices for each dataset.

    :param data: Variable number of datasets (arrays).
    :param splitsize: The size of each split (chunk).
    :return: A generator yielding tuples of slices from each dataset.
    """
    # Check that all input datasets have the same first dimension (number of samples)
    num_samples = data[0].shape[0]
    if not all(d.shape[0] == num_samples for d in data):
        raise ValueError(
            "All input datasets must have the same length along the first dimension."
        )

    num_splits = num_samples // splitsize
    remainder = num_samples % splitsize

    # Generate splits for the integer multiple parts
    for i in range(num_splits):
        start_idx = i * splitsize
        end_idx = (i + 1) * splitsize
        yield tuple(d[start_idx:end_idx] for d in data)

    # Generate split for the remainder part, if any
    if remainder > 0:
        yield tuple(d[num_splits * splitsize :] for d in data)


# --- stack result---
def get_stacked_result(
    file_path: str, s_key: str = "Signal", m_key: str = "Mask"
) -> np.ndarray:
    """
    Add up and average to get the stacked result from the HDF5 file.
    Those datasets are seperately saved in groups with keys "Signal" and "Mask", e.g. "/0_500/Signal", "/0_500/Mask".

    Args:
        file_path (str): Path to the HDF5 file.
        s_key (str): Key for the signal dataset. Defaults to "Signal".
        m_key (str): Key for the mask dataset. Defaults to "Mask".

    Returns:
        result_array (np.ndarray): The stacked result (2d array).
    """
    result = []

    def add_up(name, obj):
        if isinstance(obj, h5.Group):
            si = obj[s_key][:] # type: ignore
            mi = obj[m_key][:] # type: ignore
            s = np.ma.array(si, mask=mi)
            result.append(s)

    visititems_h5(file_path, add_up)
    result_array = np.ma.array(result)
    return result_array.mean(axis=0)
