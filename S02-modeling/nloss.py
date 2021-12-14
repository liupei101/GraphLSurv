import torch
from torch import Tensor

from utils import VERY_SMALL_NUMBER

RANK_LOSS_RATIO = 1.0

def nlog_partial_likelihood(
    y_hat: Tensor, 
    y: Tensor
) -> Tensor:
    r"""Simplified version of negative log of Breslow partial likelihood estimation.
    This is a pytorch implementation by Huang. See more in https://github.com/huangzhii/SALMON.
    
    **Note** that it only suppurts survival data with no ties (i.e., event occurrence at same time).
    
    Args:
        y_hat (Tensor): Predictions given by survival prediction model.
        y (Tensor): The absolute value of y indicates the last observed time. The sign of y 
        represents the censor status. Negative value indicates a censored example.
    """
    device = y_hat.device

    T = torch.abs(y)
    E = (y > 0).int()

    n_batch = len(T)
    R_matrix_train = torch.zeros([n_batch, n_batch], dtype=torch.int8)
    for i in range(n_batch):
        for j in range(n_batch):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = R_matrix_train.float().to(device)
    train_ystatus = E.float().to(device)

    theta = y_hat.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn

def rank_breslow_likelihood(
    y_hat: Tensor, 
    y: Tensor
) -> Tensor:
    device = y_hat.device

    T = torch.abs(y)
    E = (y > 0).int()

    n_batch = len(T)
    R_matrix_train = torch.zeros([n_batch, n_batch], dtype=torch.int8)
    R2_matrix_train = torch.zeros([n_batch, n_batch], dtype=torch.int8)
    for i in range(n_batch):
        for j in range(n_batch):
            R_matrix_train[i, j] = T[j] >= T[i]
            R2_matrix_train[i, j] = T[j] >= T[i] and E[i] == 1

    train_R = R_matrix_train.float().to(device)
    train_ystatus = E.float().to(device)

    theta = y_hat.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_cox = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    train_R2 = R2_matrix_train.float().to(device)
    new_exp_theta = torch.max(exp_theta, (1e-6 * torch.ones(n_batch)).to(device)) # prevent Nan when 1/exp_theta
    Z = torch.max(
        torch.zeros(n_batch, n_batch).float().to(device), 
        torch.ones(n_batch, n_batch).float().to(device) \
            - torch.unsqueeze(new_exp_theta, 1) * torch.unsqueeze(1 / new_exp_theta, 0)
    )
    loss_rank = torch.mean(torch.mul(Z, train_R2))
    
    loss = loss_cox + loss_rank * RANK_LOSS_RATIO

    return loss

def nlog_breslow_likelihood(
    y_hat: Tensor, 
    y: Tensor
) -> Tensor:
    r"""Complete version of negative log of Brewslow partial likelihood estimation.
    It is a generalization of Brewslow estimation, supporting survival data w/o ties.

    Args:
        y_hat (Tensor): Predictions given by survival prediction model.
        y (Tensor): The absolute value of y indicates the last observed time. The sign of y 
        represents the censor status. Negative value indicates a censored example.
    """
    # TODO
    pass


class GraphReg(object):
    """Graph Regularization Loss"""
    def __init__(self, ratio_smooth, ratio_degree, ratio_sparse):
        super(GraphReg, self).__init__()
        self.ratio_smooth = ratio_smooth
        self.ratio_degree = ratio_degree
        self.ratio_sparse = ratio_sparse

    def __call__(self, x, adj, node_mask=None, keep_batch_dim=False):
        """Calculate in batch-style
        x: [B, N_max, F]
        adj: [B, N_max, N_max]
        """
        device = x.device
        if node_mask is None:
            num_nodes = x.new_full([x.size(0)], x.size(1))
        else:
            num_nodes = node_mask.sum(-1).to(device)
        # dirichlet energy
        adj_deg = torch.sum(adj, -1, keepdim=True)
        L = adj_deg * torch.eye(adj.size(1)).unsqueeze(0).to(device) - adj
        mx = torch.matmul(x.transpose(-1, -2), torch.matmul(L, x))
        loss_dirichlet = torch.diagonal(mx, dim1=-1, dim2=-2).sum(-1) / (num_nodes * num_nodes)

        # sparse loss
        loss_degree = (node_mask * torch.log(adj_deg + VERY_SMALL_NUMBER).squeeze(-1)).sum(-1) / num_nodes
        loss_sparse = torch.pow(adj, 2).sum(dim=[-1, -2]) / (num_nodes * num_nodes)

        # loss
        loss = self.ratio_smooth * loss_dirichlet - self.ratio_degree * loss_degree + self.ratio_sparse * loss_sparse
        if keep_batch_dim:
            return loss
        else:
            return loss.mean()
