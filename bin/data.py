"""
tools for data process.

"""

import os
from typing import Generator, List, Tuple, Union

import h5py as h5
import numpy as np

# --- os ---
def get_file_dir(path:str)->str:
    """
    get the file parrent dir
    """
    return os.path.dirname(path)

def get_filename(path:str)->str:
    """
    get the file name
    """
    file_name = os.path.splitext(os.path.basename(path))[0]
    return file_name

def delete_files(file_paths:List[str])->None:
    """
    delete files
    """
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"deleted: {file_path}")
        except FileNotFoundError:
            print(f"file not found: {file_path}")
        except PermissionError:
            print(f"permission error: {file_path}")
        else:
            print(f" delete {file_path} error: {Exception}")
            
def is_exist(file_path:str)-> bool:
    """
    check file exists.
    """
    return os.path.exists(file_path)

# --- hdf5 file ---
def visititems_h5(file_path:str):
    """
    print the group and dataset info in h5 file using the visititems() method.
    """
    with h5.File(file_path, 'r') as f:
        f.visititems(print)

def del_data(file_path:str, key:str):
    """
    delete data from h5 file.
    """
    with h5.File(file_path, 'r+') as f:
        del f[key]

def read_h5(file_path:str, key:Union[str, List[str]])-> Union[np.ndarray, List[np.ndarray]]:
    """
    Read dataset from an HDF5 file.

    Args:
        filepath (str): Path to the HDF5 file.
        key (str or list): Key(s) to read from the HDF5 file.\
            It can be a single key (str) or a list of keys.

    Returns:
        out (np.ndarray or list): If a single key is provided, \
            the corresponding data is returned as a single numpy array.\
            If multiple keys are provided, a list of numpy arrays is returned.
    """    
    if not isinstance(key, (str, list)):
        raise TypeError("Key must be a string or a list of strings.")
    
    with h5.File(file_path, 'r') as f:
        avilable_keys = list(f.keys())
        
        if isinstance(key, str):
            if key not in f:
                raise ValueError(f"Key '{key}' not found. Available keys are: '{avilable_keys}'")
            else:
                return f[key][:] # type: ignore
            
        elif isinstance(key, list):
            data = []
            for k in key:
                if k not in f:
                    raise ValueError(f"Key '{k}' not found. Available keys are: '{avilable_keys}'")
                data.append(f[k][:]) # type: ignore
            
            return data
    
    return None
        
def save_h5(output:str, keys:List[str], data:list, group_name:Union[str, None]=None):
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
    with h5.File(output, 'a') as f:
        if group_name is not None:
            f.create_group(group_name)
        
        for k, d in zip(keys, data):
            if group_name is not None:
                dataset_path = f"{group_name}/{k}"
            else:
                dataset_path = k
            f.create_dataset(dataset_path, data=d)
            
# --- generator ---
def split_data_generator(data:np.ndarray, split_size:int)-> Generator[np.ndarray, None, None]:
    """
    split data by split_size.
    
    return: generator
    """
    shape = data.shape

    num_splits = shape[0] // split_size
    remaining_size = shape[0] % split_size

    for i in range(num_splits):
        start = i * split_size
        end = start + split_size
        yield data[start:end]

    if remaining_size > 0:
        start = num_splits * split_size
        end = start + remaining_size
        yield data[start:end].copy()  # Ensure a copy is made to avoid modifying the original data

def split_data_generator2(data1:np.ndarray, data2:np.ndarray, 
                          splitsize:int)->Generator[Tuple[np.ndarray, np.ndarray], None ,None]:
    """
    split 2 dataset by split_size, if first dimension is same.
    
    return: generator
    """
    
    assert data1.shape[0] == data2.shape[0], "Dimension mismatch between the two input dataset"

    num_samples = data1.shape[0]
    num_splits = num_samples // splitsize
    remainder = num_samples % splitsize

    # Generate splits for the integer multiple parts
    for i in range(num_splits):
        start_idx = i * splitsize
        end_idx = (i + 1) * splitsize
        yield data1[start_idx:end_idx], data2[start_idx:end_idx]

    # Generate split for the remainder part
    if remainder > 0:
        yield data1[num_splits*splitsize:], data2[num_splits*splitsize:]
