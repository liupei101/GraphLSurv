from typing import List, Optional, Callable, Union, Any, Tuple
import copy
import os
import os.path as osp
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import h5py
from collections.abc import Sequence

from custom_io import read_sdata, read_nfeats
from utils import auto_look_up_te

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class PatchGraphDataset(Dataset):
    r"""Dataset class of patch-based graph on survival prediction task.

    Args:
        patient_ids (list): A list of patients that are used for constructing the dataset.
        path_pat_graph (string): File path of each patient-level graph. The files must be ended with .h5.
        path_sld_feat (string): File path of each slide-level feature. The files must be ended with .h5.
        path_label (string): Path of survival data that gives `patient_id`, `t`, `e` of each 
        slide. Only support readin CSV file. 
        if_force_undirect (bool): If force return an undirect graph. Default False.
        transform (Callable): function used to transform graph, default None.
    """
    def __init__(self, patient_ids: List, path_pat_graph: str, path_sld_feat: str, path_label: str, if_force_undirect: bool = False, transform: Optional[Callable] = None):
        super(PatchGraphDataset, self).__init__()

        self.if_force_undirect = if_force_undirect
        self.transform = transform
        
        # All patient-level graph ids and its full path
        self.gids = patient_ids
        self.fpath_graphs = [osp.join(path_pat_graph, g + '.h5') for g in self.gids]
        
        # All slide-level feature ids and its full path
        self.sids = [osp.splitext(f)[0] for f in os.listdir(path_sld_feat) if osp.splitext(f)[-1] == '.h5']
        self.fpath_feats = [osp.join(path_sld_feat, s + '.h5') for s in self.sids]

        # Get labels
        sdata = read_sdata(path_label)
        self.gy, gy_mask = auto_look_up_te(sdata, self.gids, at_column='patient_id')
        if sum(gy_mask) < len(gy_mask):
            raise ValueError('Some patient ids are not found in table {}'.format(path_label))

        self._indices: Optional[Sequence] = None

        self.summary()

    def summary(self):
        print(f"Class PatchGraphDataset: if_force_undirect={self.if_force_undirect}, transform={self.transform}")

    def len(self) -> int:
        return len(self.gids)

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        path_ge = self.fpath_graphs[idx]

        with h5py.File(path_ge, 'r') as hf:
            A = hf['A'][:]
            node_indicator = [str(_, 'utf-8') for _ in hf['node_indicator'][:]]
        A = torch.from_numpy(A).to(torch.long).t().contiguous() - 1 # 0 based
        if self.if_force_undirect:
            A = to_undirected(A)

        x = []
        for slide_id in node_indicator:
            path_gn = self.fpath_feats[self.sids.index(slide_id)]
            node_feats = read_nfeats(path_gn, dtype='torch')
            x.append(node_feats)
        x = torch.cat(x, dim=0).to(torch.float)

        y = torch.Tensor([self.gy[idx]]).to(torch.float)

        data = Data(x=x, edge_index=A, y=y)

        return data

    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices
    
    @property
    def y(self) -> List:
        idxs = self.indices()
        ret_y = Tensor([self.gy[i] for i in idxs])
        return ret_y

    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0]
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")

    @property
    def num_features(self) -> int:
        r"""Alias for :py:attr:`~num_node_features`."""
        return self.num_node_features

    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        data = self[0]
        if hasattr(data, 'num_edge_features'):
            return data.num_edge_features
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_edge_features'")

    def __len__(self) -> int:
        r"""The number of examples in the dataset."""
        return len(self.indices())

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union['PatchGraphDataset', Data]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a PyTorch :obj:`LongTensor` or a :obj:`BoolTensor`, or a numpy
        :obj:`np.array`, will return a subset of the dataset at the specified
        indices."""
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)

    def index_select(self, idx: IndexType) -> 'PatchGraphDataset':
        indices = self.indices()

        if isinstance(idx, slice):
            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only integers, slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def shuffle(
        self,
        return_perm: bool = False,
    ) -> Union['PatchGraphDataset', Tuple['PatchGraphDataset', Tensor]]:
        r"""Randomly shuffles the examples in the dataset.

        Args:
            return_perm (bool, optional): If set to :obj:`True`, will return
                the random permutation used to shuffle the dataset in addition.
                (default: :obj:`False`)
        """
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        return f'{self.__class__.__name__}({arg_repr})'


