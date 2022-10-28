import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from layers import SAGPool
from layers import AnchorGraphLearner, AnchorGCN
from utils import to_dense_matrix


class GraphLSurv(torch.nn.Module):
    """
    S1: Option1, GraphLSurv 

    Args:
        args_glearner (dict): Arguments of GraphLearner contain keys of 'num_pers', 
        'ratio_anchors', 'epsilon', 'topk', 'metric_type'.
        args_gencoder (dict): Arguments of GCN encoder contain keys of 'graph_hops', 
        'dropout', 'ratio_init_graph', 'batch_norm'.
    """
    def __init__(self, in_dim, hid_dim, out_dim, num_layers=1, dropout_ratio=0.0, args_glearner=None, args_gencoder=None):
        super(GraphLSurv, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        
        self.net_glearners = nn.ModuleList()
        self.net_encoders  = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.net_glearners.append(AnchorGraphLearner(in_dim, **args_glearner))
                self.net_encoders.append(AnchorGCN(in_dim, hid_dim, **args_gencoder))
            else:
                self.net_glearners.append(AnchorGraphLearner(hid_dim, **args_glearner))
                self.net_encoders.append(AnchorGCN(hid_dim, hid_dim, **args_gencoder))

        self.lin1 = torch.nn.Linear(self.hid_dim*2, self.hid_dim)
        self.lin2 = torch.nn.Linear(self.hid_dim, self.hid_dim//2)
        self.lin3 = torch.nn.Linear(self.hid_dim//2, self.out_dim)

        self.MAX_RISK = torch.nn.Parameter(torch.tensor(5.0), requires_grad=False)

    def forward(self, data, out_anchor_graphs=None):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        init_x, init_adj, node_mask = to_dense_matrix(data, norm=True)

        prev_x = init_x
        for net_glearner, net_encoder in zip(self.net_glearners, self.net_encoders):
            # learn node-anchor adj
            node_anchor_adj, anchor_x, anchor_adj, anchor_mask = net_glearner(prev_x, node_mask)
            if out_anchor_graphs is not None:
                out_anchor_graphs['x'].append(anchor_x)
                out_anchor_graphs['adj'].append(anchor_adj)
                out_anchor_graphs['mask'].append(anchor_mask)

            # update node embedding via node-anchor-node schema
            node_vec = net_encoder(prev_x, init_adj, node_anchor_adj)

            # update x
            prev_x = node_vec

        # out_x [B, N, nhid] ---max_pool---> out [B, nhid]
        out_max = self.graph_pool(node_vec, node_mask, 'max')
        out_avg = self.graph_pool(node_vec, node_mask, 'mean')
        out = torch.cat([out_max, out_avg], dim=1)

        out = F.relu(self.lin1(out))
        out = F.dropout(out, p=self.dropout_ratio, training=self.training)
        out = F.relu(self.lin2(out))
        out = self.lin3(out)

        # prevent nan occurs in netowrk forwarding
        out = torch.where(out > self.MAX_RISK, self.MAX_RISK, out)

        return out

    @staticmethod
    def graph_pool(x, node_mask=None, pool='max'):
        """
        Shape: (batch_size, num_nodes, hidden_size)
        """
        if node_mask is None:
            node_mask = torch.ones(x.size(0) * x.size(1), dtype=torch.long, device=x.device)
        else:
            node_mask = node_mask.long().to(x.device)
        graph_embedding = scatter(x, node_mask, dim=1, reduce=pool)[:, 1, :]
        return graph_embedding

class BasicGraphConvNet(torch.nn.Module):
    """
    S3: Option3, GraphLSurv without GraphLearner
    """
    def __init__(self, in_dim, hid_dim, out_dim, hops=3, dropout_ratio=0.0):
        super(BasicGraphConvNet, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.hops = hops
        self.dropout_ratio = dropout_ratio
        self.MAX_RISK = torch.nn.Parameter(torch.tensor(5.0), requires_grad=False)

        self.graph_encoders = torch.nn.ModuleList()
        self.graph_encoders.append(GCNConv(self.in_dim, self.hid_dim))
        for _ in range(self.hops - 1):
            self.graph_encoders.append(GCNConv(self.hid_dim, self.hid_dim))

        self.lin1 = torch.nn.Linear(self.hid_dim*2, self.hid_dim)
        self.lin2 = torch.nn.Linear(self.hid_dim, self.hid_dim//2)
        self.lin3 = torch.nn.Linear(self.hid_dim//2, self.out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.hops):
            x = F.relu(self.graph_encoders[i](x, edge_index))

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        # prevent nan occurs in netowrk forwarding
        x = torch.where(x > self.MAX_RISK, self.MAX_RISK, x)

        return x

class GPNet(torch.nn.Module):
    """
    S2: Option2, Another simple GCN method with SAGPool 
    """
    def __init__(self, in_dim, hid_dim, out_dim, pooling_ratio=0.8, dropout_ratio=0.0):
        super(GPNet, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        
        self.conv1 = GCNConv(self.in_dim, self.hid_dim)
        self.pool1 = SAGPool(self.hid_dim, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.hid_dim, self.hid_dim)
        self.pool2 = SAGPool(self.hid_dim, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.hid_dim, self.hid_dim)
        self.pool3 = SAGPool(self.hid_dim, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.hid_dim*2, self.hid_dim)
        self.lin2 = torch.nn.Linear(self.hid_dim, self.hid_dim//2)
        self.lin3 = torch.nn.Linear(self.hid_dim//2, self.out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x


    