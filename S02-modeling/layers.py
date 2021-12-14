import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add

from utils import sample_anchors, compute_anchor_adj
from utils import VERY_SMALL_NUMBER

################################################
#                                              #
#                    SAGPool                   #
#                                              #
################################################

class SAGPool(torch.nn.Module):
    """
    SAGPool from https://github.com/inyeoplee77/SAGPool
    """
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x, edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm


################################################
#                                              #
#      Anchor-based Graph Learning Layer       #
#                                              #
################################################


class AnchorGraphLearner(torch.nn.Module):
    """
    Anchor-based Graph Learner.
    """
    def __init__(self, in_dim, hid_dim=128, ratio_anchors=0.2, epsilon=0.9, topk=None, metric_type='weighted_cosine'):
        super(AnchorGraphLearner, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.epsilon = epsilon
        self.topk = topk
        self.metric_type = metric_type
        self.ratio_anchors = ratio_anchors

        if self.metric_type == 'transformer':
            self.transformer = nn.Linear(in_dim, hid_dim, bias=False)
            nn.init.xavier_uniform_(self.transformer.weight)
            print("Transformer-based GraphLearner with hid_dim = {}".format(hid_dim))
        else:
            raise NotImplementedError('{} has not been implemented.'.format(self.metric_type))

        self.summary()

    def summary(self):
        print(f'[class GraphLearner] learning method: {self.metric_type}, in_dim: {self.in_dim}, hid_dim: {self.hid_dim}, \
            epsilon: {self.epsilon}, topk: {self.topk}, ratio_anchors: {self.ratio_anchors}')

    def forward(self, x, node_mask):
        """
        x: Tensor with shape of [B, N_max, F], F denotes feature dimension.
        node_mask: BoolTensor with shape of [B, N_sum].
        """
        # random sample x anchors = [B, N_max, F], N_max = N_max * ratio_anchors
        anchors_x, anchor_mask = sample_anchors(x, node_mask, self.ratio_anchors)

        if self.metric_type == 'transformer':
            context_fc = self.transformer(x)
            context_norm = F.normalize(context_fc, p=2, dim=-1)

            anchors_fc = self.transformer(anchors_x)
            anchors_norm = F.normalize(anchors_fc, p=2, dim=-1)

            attention = torch.matmul(context_norm, anchors_norm.transpose(-1, -2))
            markoff_value = 0

        if self.epsilon is not None:
            attention = build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = build_epsilon_neighbourhood(attention, self.topk, markoff_value)

        anchor_adj = compute_anchor_adj(attention, anchor_mask=anchor_mask)

        return attention, anchors_x, anchor_adj, anchor_mask

def build_epsilon_neighbourhood(attention, epsilon, markoff_value):
    """
    Node-Anchor attention = [B, N, S] with fake nodes and anchors
    """
    mask = (attention > epsilon).detach().float()
    weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
    return weighted_adjacency_matrix

def build_knn_neighbourhood(attention, topk, markoff_value):
    """
    Node-Anchor attention = [B, N, S] with fake nodes and anchors
    """
    topk = min(topk, attention.size(-1))
    knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
    weighted_adjacency_matrix = attention.new_full(attention.size(), markoff_value).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


################################################
#                                              #
# Anchor-based Graph Convolution Network Layer #
#                                              #
################################################

class AnchorGCNLayer(nn.Module):
    """
    Simple AnchorGCN layer, similar to https://arxiv.org/abs/1609.02907 and 
    https://arxiv.org/abs/2006.13009.
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(AnchorGCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, anchor_mp=True, batch_norm=True):
        support = torch.matmul(input, self.weight)

        if anchor_mp:
            node_anchor_adj = adj
            node_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-2, keepdim=True), min=VERY_SMALL_NUMBER)
            anchor_norm = node_anchor_adj / torch.clamp(torch.sum(node_anchor_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)
            output = torch.matmul(anchor_norm, torch.matmul(node_norm.transpose(-1, -2), support))
        else:
            node_adj = adj
            output = torch.matmul(node_adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class AnchorGCN(nn.Module):
    """
    A GCN model with defined function 'hybrid_message_passing'.

    Anchor-based strategy is introduced to optimize time and space consuming.
    See more in https://arxiv.org/abs/1609.02907 and https://arxiv.org/abs/2006.13009.
    """

    def __init__(self, nfeat, nhid, graph_hops=2, ratio_init_graph=0.0, dropout_ratio=None, batch_norm=False):
        super(AnchorGCN, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.ratio_init_graph = ratio_init_graph

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(AnchorGCNLayer(nfeat, nhid, batch_norm=batch_norm))
        for _ in range(graph_hops - 1):
            self.graph_encoders.append(AnchorGCNLayer(nhid, nhid, batch_norm=batch_norm))

    def forward(self, x, init_adj, node_anchor_adj):
        for encoder in self.graph_encoders[:-1]:
            x = self.hybrid_message_passing(encoder, x, init_adj, node_anchor_adj)
        out_x = self.hybrid_message_passing(self.graph_encoders[-1], x, init_adj, node_anchor_adj, return_raw=True)

        return out_x

    def hybrid_message_passing(self, encoder, x, init_adj, node_anchor_adj, return_raw=False):
        """
        init_adj: init_adj must be normalized.
        """
        x_from_init_graph = encoder(x, init_adj, anchor_mp=False, batch_norm=False)
        x_from_learned_graph = encoder(x, node_anchor_adj, anchor_mp=True, batch_norm=False)
        x = self.hybrid_updata_x(x_from_init_graph, x_from_learned_graph)

        if return_raw:
            return x

        if encoder.bn is not None:
            x = encoder.compute_bn(x)
        x = torch.relu(x)
        x = F.dropout(x, self.dropout_ratio, training=self.training)

        return x
    
    def hybrid_updata_x(self, x_init_graph, x_new_graph):
        return self.ratio_init_graph * x_init_graph + (1 - self.ratio_init_graph) * x_new_graph


