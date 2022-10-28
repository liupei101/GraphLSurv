from typing import List, Optional, Callable, Union, Any, Tuple

import sys
import pandas as pd
from pandas import DataFrame
import torch
from torch import Tensor
from torch_scatter import scatter_add
import numpy as np
from numpy import ndarray
import copy
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import add_remaining_self_loops
from lifelines.utils import concordance_index as ci

VERY_SMALL_NUMBER = 1e-12
SMALL_NUMERIC_FLOAT = 1e-5

def check_sdata(df: DataFrame) -> bool:
    r"""Check if data is a standard survival data for pathology.

    Args:
        df (pandas.DataFrame): Data to check.
    """
    COLUMNS_IN_SURV = ['patient_id', 'pathology_id', 'e', 't']
    return all([c in df.columns for c in COLUMNS_IN_SURV])

def sdata_on_pt(
    df: DataFrame, 
    pt: Union[float, int],
    column_e: Optional[str] = 'e', 
    column_t: Optional[str] = 't'
)-> DataFrame:
    r"""Cut survival data on a specified time point, i.e. the study 
    endpoint.
    """
    if column_t is None:
        column_t = 't'
    if column_e is None:
        column_e = 'e'

    e, t = [], []
    for i in df.index:
        if df.loc[i, column_t] <= pt:
            ne, nt = df.loc[i, column_e], df.loc[i, column_t]
        else:
            ne, nt = 0, pt
        e.append(ne)
        t.append(nt)

    df.loc[:, column_e] = e
    df.loc[:, column_t] = t

    return df

def auto_look_up_te(
    df: DataFrame, 
    ids: List[str],
    at_column: Optional[str] = None
) -> Union[List[float], List[int]]:
    r"""Look the label of given `patient_id` or `pathology_id`.
    if not specified the column to look up, the function will automatically 
    infer the column.

    [Notice] If it is a censored example, the return label is negative, if it is an 
    event example, the return label is positive.
    """
    if at_column is None:
        if df['patient_id'].isin(ids).sum() >= len(ids):
            # print("Matched `ids` in column `patient_id`")
            at_column = 'patient_id'

        elif df['pathology_id'].isin(ids).sum() >= len(ids):
            # print("Matched `ids` in column `pathology_id`")
            at_column = 'pathology_id'
        else:
            raise RuntimeError('Some `ids` not matched')

    res, mask = [], []
    for cid in ids:
        matched_df = df[df[at_column] == cid]
        if matched_df.empty:
            res.append(0)
            mask.append(0)
        else:
            irow = matched_df.index[0]
            if pd.isnull(df.loc[irow, 't']) or pd.isnull(df.loc[irow, 'e']):
                res.append(0)
                mask.append(0)
            else:
                # Negetive time indicates a right-censored one
                x = -df.loc[irow, 't'] if df.loc[irow, 'e'] == 0 else df.loc[irow, 't']
                res.append(x)
                mask.append(1)
    
    return res, mask

def concordance_index(
    y_pred: Union[Tensor, ndarray],
    y_true: Union[Tensor, ndarray] 
) -> float:
    """Compute the concordance-index value.

    Args:
        y_pred (Union[Tensor, ndarray]): Predicted value.
        y_true (Union[Tensor, ndarray]): Observed time. Negative values are considered right censored.
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    t = np.abs(y_true)
    e = (y_true > 0).astype(np.int32)

    return ci(t, -y_pred, e)

def eval_sdata(
    y_hat: Union[Tensor, ndarray], 
    y: Union[Tensor, ndarray],
    loss: Optional[Union[str, Callable]] = None
) -> Union[Tuple, float]:
    r"""Evaluation on 'y' from survival data and 'y_hat' from model prediction.

    Args:
        y_hat (Union[Tensor, ndarray]): Predictions given by survival prediction model.
        y (Union[Tensor, ndarray]): The absolute value of y indicates the last observed time. 
        The sign of y represents the censor status. Negative value indicates a censored example.
        loss (Union[str, Callable]): If loss is None, then only return the c_index; else return the specified 
        loss function of y and y_hat.
    """
    val_ci = concordance_index(y_hat, y)

    if loss is None:
        return val_ci
    elif callable(loss):
        return loss(y_hat, y)
    else:
        from nloss import nlog_partial_likelihood, nlog_breslow_likelihood
        from nloss import rank_breslow_likelihood
        assert loss in ['sim-breslow', 'breslow', 'rank-breslow']
        if isinstance(y_hat, ndarray):
            y_hat = torch.from_numpy(y_hat)
        if isinstance(y, ndarray):
            y = torch.from_numpy(y)

        if loss == 'sim-breslow':
            val_loss = nlog_partial_likelihood(y_hat, y)
        elif loss == 'breslow':
            val_loss = nlog_breslow_likelihood(y_hat, y)
        elif loss == 'rank-breslow':
            val_loss = rank_breslow_likelihood(y_hat, y)
        else:
            raise NotImplementedError('loss {} has not been implemented.'.format(loss))

        return val_ci, val_loss

def utils_avg(L):
    return 1.0 * sum(L) / len(L)

def collect_tensor(collector, y_hat, y):
    if collector['y_hat'] is None:
        collector['y_hat'] = y_hat
    else:
        collector['y_hat'] = torch.cat([collector['y_hat'], y_hat], dim=0)
        
    if collector['y'] is None:
        collector['y'] = y
    else:
        collector['y'] = torch.cat([collector['y'], y])
    
    return collector

def print_config(config, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL CONFIGURATION ****************", file=f)
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val), file=f)
    print("**************** MODEL CONFIGURATION ****************", file=f)
    
    if print_to_path is not None:
        f.close()

def print_metrics(metrics, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL METRICS ****************", file=f)
    for key in sorted(metrics.keys()):
        val = metrics[key]
        for v in val:
            cur_key = key + '/' + v[0]
            keystr  = "{}".format(cur_key) + (" " * (20 - len(cur_key)))
            valstr  = "{}".format(v[1])
            if isinstance(v[1], list):
                valstr = "{}, avg/std = {:.5f}/{:.5f}".format(valstr, np.mean(v[1]), np.std(v[1]))
            print("{} -->   {}".format(keystr, valstr), file=f)
    print("**************** MODEL METRICS ****************", file=f)
    
    if print_to_path is not None:
        f.close()

######################################################
#                                                    #
#      Data splition Generic Functions               #
#                                                    #
######################################################

class DataSplit(object):
    """docstring for DataSplit"""
    def __init__(self, exp_type, test_ratio=0.2, val_ratio=0.2, kfold=5):
        super(DataSplit, self).__init__()
        self.exp_type = exp_type
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.kfold = kfold
        assert exp_type in ['sim', 'std-hpopt', 'std-perfm']

    def __call__(self, n, stratify=None):
        if stratify is None:
            return self.split_perm(n)
        else:
            if isinstance(stratify, Tensor):
                stratify = stratify.numpy()
            else:
                stratify = np.array(stratify)
            
            y0_idx   = np.where(stratify == 0)[0]
            y0_split = self.split_perm(y0_idx.shape[0])
            y1_idx   = np.where(stratify == 1)[0]
            y1_split = self.split_perm(y1_idx.shape[0])

            if self.exp_type == 'sim':
                ret = []
                for i in range(3):
                    cur_idx = np.append(y0_idx[y0_split[i]], y1_idx[y1_split[i]])
                    np.random.shuffle(cur_idx)
                    ret.append(cur_idx)
                return ret
            elif self.exp_type == 'std-perfm':
                ret = []
                for i in range(2):
                    cur_idx = np.append(y0_idx[y0_split[i]], y1_idx[y1_split[i]])
                    np.random.shuffle(cur_idx)
                    ret.append(cur_idx)
                return ret
            elif self.exp_type == 'std-hpopt':
                res_fold = []
                for i in range(self.kfold):
                    fold_idx = []
                    for j in range(2):
                        cur_idx = np.append(y0_idx[y0_split[i][j]], y1_idx[y1_split[i][j]])
                        np.random.shuffle(cur_idx)
                        fold_idx.append(cur_idx)
                    res_fold.append(fold_idx)
                return res_fold

    def split_perm(self, n):
        nperm = np.random.permutation(n)
        if self.exp_type == 'sim':
            n_train = int(np.around(n * (1 - self.test_ratio)) + SMALL_NUMERIC_FLOAT)
            n_test  = n - n_train
            n_val   = int(np.around(n_train * self.val_ratio) + SMALL_NUMERIC_FLOAT)
            n_train = n_train - n_val
            return nperm[:n_train], nperm[n_train:(n_train+n_val)], nperm[(n_train+n_val):]
        elif self.exp_type == 'std-perfm':
            n_train = int(np.around(n * (1 - self.test_ratio)) + SMALL_NUMERIC_FLOAT)
            n_test  = n - n_train
            return nperm[:n_train], nperm[n_train:]
        elif self.exp_type == 'std-hpopt':
            n_train = int(np.around(n * (1 - self.test_ratio)) + SMALL_NUMERIC_FLOAT)
            n_fold = int(np.ceil(n_train / self.kfold) + SMALL_NUMERIC_FLOAT)
            res_fold = []
            for i in range(self.kfold):
                fold_val_idx = nperm[(i*n_fold):min(n_train, (i+1)*n_fold)]
                fold_train_idx = np.append(nperm[:(i*n_fold)], nperm[min(n_train, (i+1)*n_fold):n_train])
                res_fold.append([fold_train_idx, fold_val_idx])
            return res_fold

######################################################
#                                                    #
#      Graph Related Generic Functions               #
#                                                    #
######################################################

def to_dense_matrix(data, norm=False, self_loop=True):
    """
    data: torch_geometric.data.Data
    """
    x, edge_index, batch = data.x, data.edge_index, data.batch
    # add self-loops
    if self_loop:
        edge_index, _ = add_remaining_self_loops(edge_index)
    # init_node_vec = [B, N_max, F], N_max = Max_i(N_i) with fake nodes
    init_x, node_mask = to_dense_batch(x, batch)
    # init_adj = [B, N_max, N_max]
    init_adj = to_dense_adj(edge_index, batch).byte()
    if norm:
        init_adj = batch_normalize_adj(init_adj, mask=node_mask)

    return init_x, init_adj, node_mask

def sample_anchors(x, node_mask, ratio, fill_value=0):
    """
    x: [B, N_max, F]
    node_mask: [B, N_max]
    """
    batch_size = x.size(0)
    num_nodes = node_mask.sum(1)
    max_nodes = num_nodes.max().cpu()
    assert max_nodes == x.size(1)

    sampled_num_nodes = (ratio * num_nodes).to(num_nodes.dtype)
    sampled_max_nodes = sampled_num_nodes.max().cpu()
    sampled_col_index = [torch.randperm(num_nodes[i])[:sampled_num_nodes[i]] for i in range(x.size(0))]
    sampled_idx = torch.cat([i*max_nodes + sampled_col_index[i] for i in range(x.size(0))])
    insert_idx = torch.cat([i*sampled_max_nodes + torch.arange(sampled_num_nodes[i]) for i in range(x.size(0))])

    x = x.view([x.size(0) * x.size(1)] + list(x.size())[2:])
    size = [batch_size * sampled_max_nodes] + list(x.size())[1:]
    
    out = x.new_full(size, fill_value)
    out[insert_idx] = x[sampled_idx]
    out = out.view([batch_size, sampled_max_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * sampled_max_nodes, dtype=torch.bool, device=x.device)
    mask[insert_idx] = 1
    mask = mask.view([batch_size, sampled_max_nodes])

    return out, mask

def batch_normalize_adj(mx, mask=None):
    """
    Row-normalize matrix: symmetric normalized Laplacian.
    mx = [batch_size, N, N]
    """
    mx = mx.float()
    rowsum = torch.clamp(mx.sum(-1), min=1e-12)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    if mask is not None:
        r_inv_sqrt = r_inv_sqrt * mask
        
    return r_inv_sqrt.unsqueeze(2) * mx * r_inv_sqrt.unsqueeze(1)

def compute_anchor_adj(node_anchor_adj, anchor_mask=None):
    node_norm = node_anchor_adj / torch.clamp(node_anchor_adj.sum(dim=-2, keepdim=True), min=VERY_SMALL_NUMBER)
    anchor_norm = node_anchor_adj / torch.clamp(node_anchor_adj.sum(dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
    anchor_adj = torch.matmul(node_norm.transpose(-1, -2), anchor_norm)

    markoff_value = 0
    if anchor_mask is not None:
        anchor_adj = anchor_adj.masked_fill_(~anchor_mask.bool().unsqueeze(-1), markoff_value)
        anchor_adj = anchor_adj.masked_fill_(~anchor_mask.bool().unsqueeze(-2), markoff_value)

    return anchor_adj

def dense_to_sparse(A, batch):
    batch_size = int(batch.max()) + 1
    num_nodes = scatter_add(batch.new_ones(batch.size(0)), batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    index = A.nonzero(as_tuple=True)
    edge_attr = A[index]
    batch_st = cum_nodes[index[0]]
    index = (batch_st + index[1], batch_st + index[2])

    return torch.stack(index, dim=0), edge_attr


######################################################
#                                                    #
#      Arguments Related Generic Functions           #
#                                                    #
######################################################

def get_args_agl_layer(cfg):
    args = {
        'hid_dim': cfg['agl_hid_dim'], 
        'ratio_anchors': cfg['agl_ratio_anchors'],
        'epsilon': cfg['agl_epsilon'],
        'topk': cfg['agl_topk'],
        'metric_type': cfg['agl_metric_type']
    }
    return args

def get_args_age_layer(cfg):
    args = {
        'graph_hops': cfg['age_graph_hops'], 
        'ratio_init_graph': cfg['age_ratio_init_graph'],
        'dropout_ratio': cfg['age_dropout_ratio'],
        'batch_norm': cfg['age_batch_norm']
    }
    return args
