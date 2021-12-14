from typing import List, Optional, Callable, Union, Any, Tuple

import os.path as osp
import numpy as np
import math
import pandas as pd
import argparse
import h5py
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph

import sys
sys.path.append("..") 
from custom_io import save_hdf5


class GraphBuilder(object):
    r"""Base class of graph builder

    Args:
        root (string): Directory of features of slide patches.
        sids (list): Slide ids used to build the graph.
        verbose (bool): if print class summary.
    """
    def __init__(self, root: str, sids: Optional[List[str]] = None, verbose: bool = False):
        assert len(sids) > 0

        self.verbose = verbose
        self.sids = []
        self.feats = None
        self.edges = None
        self.method = None

        self.feats = []
        self.coords = []
        INF = -1
        for sid in sids:
            path = osp.join(root, sid + '.h5')
            if not osp.exists(path):
                print("[Warning] Cannot find h5 files of slide %s." % sid)
                continue

            self.sids.append(sid)
            with h5py.File(path, 'r') as hf:
                self.feats.append(hf['features'][:])
                self.coords.append(hf['coords'][:])
                INF = max(INF, np.max(self.coords[-1]))
        # preprocess coordinates for avoiding overlapping
        BIAS = np.array([INF, INF], dtype=np.float32)
        for i in range(len(self.coords)-1):
            self.coords[i+1] = self.coords[i+1] + (i+1)*BIAS

        self.sids = np.array(self.sids, dtype='S')
        self.feats = np.concatenate(self.feats, axis=0)
        self.coords = np.concatenate(self.coords, axis=0)

        if self.verbose:
            self.summary()

    def summary(self):
        print("GraphBuilder Summary:")
        print(f"Graph with {len(self.feats)} nodes, {len(self.edges) if self.edges is not None else 'NA'} edges.")
        print(f"Each node has {self.feats.shape[1]} features.")
        print(f"Building topology with method {self.method if self.method is not None else 'NA'}.")

    def build(self, 
        method: Optional[Union[Callable, str]] = 'knn',
        **kwargs
    ):
        if method is None:
            method = 'knn'
        self.method = method

        if isinstance(method, str):
            if method == 'knn':
                k, thresh = kwargs['k'], kwargs['threshold']
                adj = kneighbors_graph(
                    self.feats,
                    k,
                    mode="distance",
                    include_self=False,
                    metric="euclidean").toarray()
                # filter edges that are too far (ie larger than thresh)
                if thresh is not None:
                    adj[adj > thresh] = 0
            elif method == 'radius':
                r = kwargs['r']
                adj = radius_neighbors_graph(
                    self.coords,
                    r,
                    mode="distance",
                    include_self=False,
                    metric='euclidean').toarray()
            else:
                raise NotImplementedError('method not recognized.')
        else:
            adj = method(self.feats, kwargs)

        nodes_from, nodes_to = np.nonzero(adj)
        nodes_from = np.expand_dims(nodes_from, axis=-1)
        nodes_to = np.expand_dims(nodes_to, axis=-1)
        # Index of node are 1 based
        self.edges = np.concatenate((nodes_from, nodes_to), axis=1) + 1

        if self.verbose:
            self.summary()

    def return_graph(self) -> dict:
        graph = {'A': self.edges, 'node_indicator': self.sids, 'X': self.feats}
        return graph

    def save(self, path: str, with_feats: bool = False) -> str:
        if with_feats:
            asset_dicts = {'A': self.edges, 'node_indicator': self.sids, 'X': self.feats}
        else:
            asset_dict = {'A': self.edges, 'node_indicator': self.sids}
        save_hdf5(path, asset_dict, mode='w')
        return path

def parse_args():
    parser = argparse.ArgumentParser(description='Tools of Building Graph Topology')
    parser.add_argument('--dir_input', type=str, default=None, 
        help='Directory of slide patch features that are saved as H5 files.')
    parser.add_argument('--dir_output', type=str, default=None,
        help='Directory of slide graph topology that are saved as H5 files.')
    parser.add_argument('--csv_sld2pat', type=str, default=None,
        help='CSV file used to mapping `slide_id` to `patient_id`.')
    parser.add_argument('--graph_level', type=str, default='patient',
        help='Building graph on which level, default on patient level.')
    parser.add_argument('--method', type=str, default='knn',
        help='Building graph using method, default KNN.')
    parser.add_argument('--num_neighbours', '-k', type=int, default=6,
        help='k nearest neighbors.')
    parser.add_argument('--threshold', '-t', type=int, default=50,
        help='The edge whose distance larger than t will be removed from graph.')
    parser.add_argument('--radius', '-r', type=str, default='diag',
        help='radius for method `radius_neighbors_graph`.')
    parser.add_argument('--num_workers', type=int, default=1,
        help='Number of cpus to process the whole dataset.')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
        help='if verbose.')

    return parser.parse_args()

def pipeline(gid, gval, args):
    if not any([osp.exists(osp.join(args.dir_input, _ + '.h5')) for _ in gval]):
        return ''

    save_path = osp.join(args.dir_output, gid + '.h5')
    graph_builder = GraphBuilder(args.dir_input, gval, verbose=args.verbose)
    if args.method == 'knn':
        graph_builder.build(method='knn', k=args.num_neighbours, threshold=args.threshold)
    elif args.method == 'radius':
        graph_builder.build(method='radius', r=args.radius_value)
    graph_builder.save(save_path)

    return save_path

def main(args):
    import time
    import os
    import multiprocessing as mp
    from functools import partial

    # inspect args
    if not osp.exists(args.dir_output):
        os.makedirs(args.dir_output)
    if args.threshold is not None and args.threshold <= 0:
        args.threshold = None
    assert args.radius in ['one', 'diag', 'double']
    if args.radius == 'one':
        args.radius_value = 1024.0
    elif args.radius == 'diag':
        args.radius_value = 1024.0 * math.sqrt(2.0)
    elif args.radius == 'double':
        args.radius_value = 1024.0 * 2
    args.radius_value += 0.5
    print(args)

    sld2pat = pd.read_csv(args.csv_sld2pat, dtype={"patient_id": str, 'pathology_id': str})

    at_column = args.graph_level + '_id'
    at_groups = sld2pat.groupby(at_column).groups

    gids = [_ for _ in at_groups.keys()]
    gvals = [list(sld2pat.loc[at_groups[gid], 'pathology_id']) for gid in gids]

    func_pipeline = partial(pipeline, args=args)
    res_process = []
    if args.num_workers > 1:
        with mp.Pool(processes=args.num_workers) as pool:
            res_process = pool.starmap(func_pipeline, zip(gids, gvals))
    else:
        for i in range(len(gids)):
            tstart = time.time()
            res = func_pipeline(gids[i], gvals[i])
            if res != '':
                print("processed %s, took %.5f seconds." % (gids[i], time.time() - tstart))
            res_process.append(res)

    print("Processed %d graphs." % len([_ for _ in res_process if _ != '']))

if __name__ == '__main__':
    args = parse_args()
    main(args)

