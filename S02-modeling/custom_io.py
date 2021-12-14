from typing import List, Optional, Callable, Union, Any, Tuple

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import os.path as osp
import h5py
import torch
from torch import Tensor

from utils import check_sdata, sdata_on_pt

def read_sdata(path: str, pt: Optional[Union[float, int]] = None) -> DataFrame:
    r"""Read survival data including event, time and identifier.

    Args:
        path (string): Path for reading survival data.
    """
    assert osp.splitext(path)[-1] == '.csv', "Not support CSV format files."

    df_sur = pd.read_csv(path, dtype={'pathology_id': str, 'patient_id': str})
    assert check_sdata(df_sur), 'Please check if data contains \
    columns of patient_id, pathology_id, e, t'

    if pt is not None:
        df_sur = sdata_on_pt(df_sur, pt)

    return df_sur

def read_gpath(path_graph: dict) -> Tuple:
    r"""Read file path for building graph topology.

    Args:
        path_graph (dict): Path of files, such as node features and edges.
    """
    assert 'node' in path_graph.keys() and 'edge' in path_graph.keys(), "Please check \
    if there are keys of `node` and `edge` in `path_graph`"

    # read paths of graph edges
    paths_ge = [osp.join(path_graph['edge'], f) for f in os.listdir(path_graph['edge']) 
        if osp.splitext(f)[-1] == '.h5']
    gids = [osp.splitext(osp.split(p)[-1])[0] for p in paths_ge]

    # read paths of graph node features
    paths_gn = [osp.join(path_graph['node'], f) for f in os.listdir(path_graph['node']) 
        if osp.splitext(f)[-1] == '.h5']
    sids = [osp.splitext(osp.split(p)[-1])[0] for p in paths_gn]

    return gids, paths_ge, sids, paths_gn

def read_nfeats(path: str, dtype: str = 'torch') -> Union[Tensor, np.ndarray]:
    r"""Read node features from path.

    Args:
        path (string): Read data from path.
        dtype (string): Type of return data, default `torch`.
    """
    assert dtype in ['numpy', 'torch']

    with h5py.File(path, 'r') as hf:
        nfeats = hf['features'][:]

    if dtype == 'numpy':
        return nfeats
    else:
        return torch.from_numpy(nfeats)

def save_hdf5(
    output_path: str, 
    asset_dict: dict, 
    attr_dict: Optional[dict] = None, 
    mode: str = 'a'
) -> str:
    r"""Utility function for saving a HDF5 file.

    Args:
        output_path (string): Write data from output_path.
        asset_dict (dict): Dataset to save.
        attr_dict (dict): Dataset attributes to save.
        mode (string): options of 'a' and 'w'.
    """
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def save_surv_prediction(y_pred, y_true, save_path):
    r"""Surival prediction saving.

    Args:
        y_pred (Tensor or ndarray): predicted values.
        y_true (Tensor or ndarray): true labels.
        save_path (string): path for saving.
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()

    y_pred = np.squeeze(y_pred)
    y_true = np.squeeze(y_true)

    df = pd.DataFrame({'pred': y_pred, 'true': y_true})
    df.to_csv(save_path, index=False)

def read_datasplit_npz(path: str) -> Tuple:
    data_npz = np.load(path)
    
    pids_train = [str(s) for s in data_npz['train_patients']]
    pids_val   = [str(s) for s in data_npz['val_patients']]
    pids_test  = [str(s) for s in data_npz['test_patients']]

    return pids_train, pids_val, pids_test

