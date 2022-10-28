import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch import Tensor
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

import dataset as ds
from nets import GPNet, GraphLSurv, BasicGraphConvNet
from nloss import GraphReg
from utils import DataSplit
from utils import eval_sdata
from utils import utils_avg, collect_tensor
from utils import print_metrics, print_config
from utils import get_args_agl_layer, get_args_age_layer
from nloss import nlog_partial_likelihood, nlog_breslow_likelihood
from nloss import rank_breslow_likelihood
from custom_io import save_surv_prediction, read_datasplit_npz


class DRPMHandler(object):
    """Deep Risk Predition Model Handler.
    Handler the model train/val/test for:
    S1. Option1: GraphLSurv
    S2. Option2: Hierarchical GCN
    S3. Option3: BasicGraphConvNet, i.e., GraphLSurv without GraphLearner
    """
    def __init__(self, cfg):
        # cuda
        if not cfg['no_cuda'] and torch.cuda.is_available():
            cfg['device'] = 'cuda' if cfg['cuda_id'] < 0 else 'cuda:%d' % cfg['cuda_id']
        else:
            cfg['device'] = 'cpu'
        print('[Using {}]'.format(cfg['device']))

        # random state (may still be random due to 'shuffle=True' in trainloader)
        seed = cfg['random_state']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if 'cuda' in cfg['device']:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        # output
        self.writer = SummaryWriter(cfg['save_path'])
        self.last_ckpt_path = osp.join(cfg['save_path'], 'model-last.pth')
        self.best_ckpt_path = osp.join(cfg['save_path'], 'model-best.pth')
        self.metrics_path   = osp.join(cfg['save_path'], 'metrics.txt')
        self.config_path    = osp.join(cfg['save_path'], 'print_config.txt')

        # model
        if cfg['task'] == 'GraphLSurv':
            self.model = GraphLSurv(
                cfg['model_in_dim'], cfg['model_hid_dim'], cfg['model_out_dim'], 
                num_layers=cfg['model_num_layers'],
                dropout_ratio=cfg['model_dropout_ratio'],
                args_glearner=get_args_agl_layer(cfg),
                args_gencoder=get_args_age_layer(cfg))
        elif cfg['task'] == 'GraphPredict':
            self.model = GPNet(
                cfg['model_in_dim'], cfg['model_hid_dim'], cfg['model_out_dim'], 
                pooling_ratio=cfg['model_pooling_ratio'],
                dropout_ratio=cfg['model_dropout_ratio'])
        elif cfg['task'] == 'GraphBasic':
            self.model = BasicGraphConvNet(
                cfg['model_in_dim'], cfg['model_hid_dim'], cfg['model_out_dim'], 
                hops=cfg['model_hops'],
                dropout_ratio=cfg['model_dropout_ratio'])
        else:
            raise ValueError(f"Expected values of GraphLSurv, GraphPredict or GraphBasic, but got {cfg['task']}")
        self.model = self.model.to(cfg['device'])

        # loss
        self.loss = self._loss(cfg['loss_type'])
        self.loss_reg  = self._loss_reg(cfg['reg_l1'])
        if cfg['task'] == 'GraphLSurv' and cfg['graph_reg']:
            self.loss_graph = self._loss_reg_graph(cfg['smoothness_ratio'], cfg['sparsity_ratio'], cfg['degree_ratio'])
        else:
            self.loss_graph = None

        # Optimizer
        self.optimizer = self._optimizier(cfg['optimizer_type'], self.model, lr=cfg['lr'])

        # LR scheduler (watch val_loss to update, or watch train_loss to update if not specify validation set)
        self.steplr = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=cfg['lr_factor'], 
            patience=cfg['lr_patience'], min_lr=cfg['lr_min'], verbose=True)

        self.cfg = cfg
        print_config(cfg, print_to_path=self.config_path)

    def exec(self):
        task = self.cfg['task']
        experiment = self.cfg['experiment_type']
        print('[INFO] start {} experiment-{}.'.format(task, experiment))

        path_split = self.cfg['data_split'].format(self.cfg['seed_split'])
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        print('[INFO] loaded patient ids from {}'.format(path_split))
        
        # For reporting results
        if experiment == 'sim':
            # Prepare datasets 
            train_set = self._prepare_dataset(pids_train, self.cfg)
            val_set   = self._prepare_dataset(pids_val, self.cfg)
            test_set  = self._prepare_dataset(pids_test, self.cfg)
            train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], num_workers=self.cfg['num_workers'], shuffle=True)
            val_loader   = DataLoader(val_set,   batch_size=self.cfg['batch_size'], num_workers=self.cfg['num_workers'], shuffle=False)
            test_loader  = DataLoader(test_set,  batch_size=self.cfg['batch_size'], num_workers=self.cfg['num_workers'], shuffle=False)
            
            # Train
            val_name = 'validation'
            val_loaders = {'validation': val_loader, 'test': test_loader}
            self._run_training(train_loader, val_loaders=val_loaders, val_name=val_name, measure=True, save=True)

            # Evals
            metrics = dict()
            evals_loader = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
            for k, loader in evals_loader.items():
                cltor = self.test_model(self.model, loader, checkpoint=self.best_ckpt_path, device=self.cfg['device'])
                ci, loss = eval_sdata(cltor['y_hat'], cltor['y'], loss=self.cfg['loss_type'])
                metrics[k] = [('cindex', ci), ('loss', loss)]

                if self.cfg['save_prediction']:
                    path_save_pred = osp.join(self.cfg['save_path'], 'pred_surv_{}.csv'.format(k))
                    save_surv_prediction(cltor['y_hat'], cltor['y'], path_save_pred)
        # For other experiments, such as 5-fold cross-validation.
        else:
            # Data splitting function for 'std-perfm' and 'std-hpopt'
            spliter = DataSplit(experiment)

            all_pids = pids_train + pids_val + pids_test
            full_data = self._prepare_dataset(all_pids, self.cfg)
            label = (full_data.y > 0).int() if self.cfg['split_stratify'] else None
            
            if experiment == 'std-perfm':
                # Data
                train_idx, test_idx = spliter(len(full_data), stratify=label)
                train_set = Subset(full_data, train_idx)
                test_set  = Subset(full_data, test_idx)
                train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], num_workers=self.cfg['num_workers'], shuffle=True)
                test_loader  = DataLoader(test_set,  batch_size=self.cfg['batch_size'], num_workers=self.cfg['num_workers'], shuffle=False)

                # Train
                self._run_training(train_loader, measure=True, save=True)

                # Evals
                test_cltor = self.test_model(self.model, test_loader, checkpoint=self.last_ckpt_path, device=self.cfg['device'])
                ci, loss = eval_sdata(test_cltor['y_hat'], test_cltor['y'], loss=self.cfg['loss_type'])
                metrics = {'test': [('cindex', ci), ('loss', loss)]}

            elif experiment == 'std-hpopt':
                # Data
                folds_ci   = []
                folds_loss = []

                folds_idx  = spliter(len(full_data), stratify=label)
                for i in range(len(folds_idx)):
                    print("[INFO] std-hpopt in fold {}".format(i + 1))
                    # Data
                    fdx = folds_idx[i]
                    train_loader = DataLoader(Subset(full_data, fdx[0]), batch_size=self.cfg['batch_size'], num_workers=self.cfg['num_workers'], shuffle=True)
                    val_loader   = DataLoader(Subset(full_data, fdx[1]), batch_size=self.cfg['batch_size'], num_workers=self.cfg['num_workers'], shuffle=False)
                    
                    # Train
                    self._run_training(train_loader, measure=True, save=False)
                    
                    # Evals
                    cltor = self.test_model(self.model, val_loader, device=self.cfg['device'])
                    ci, loss = eval_sdata(cltor['y_hat'], cltor['y'], loss=self.cfg['loss_type'])

                    folds_ci.append(ci)
                    folds_loss.append(loss)

                metrics = {'cross_val': [('cindex', folds_ci), ('loss', folds_loss)]}

        print_metrics(metrics, print_to_path=self.metrics_path)

        return metrics

    def _run_training(self, train_loader, val_loaders=None, val_name=None, measure=True, save=True, **kws):
        """Traing the GCN model.

        Args:
            train_loader ('DataLoader'): DatasetLoader of training set.
            val_loaders (dict): A dict like {'val': loader1, 'test': loader2}, gives the datasets
            to evaluate at each epoch.
            val_name (string): The dataset used to perform early stopping and optimal model saving.
            measure (bool): If measure training set at each epoch.
        """
        epochs = self.cfg['epochs']
        assert self.cfg['bp_every_iters'] % self.cfg['batch_size'] == 0, "Batch size must be divided by bp_every_iters."
        if val_name is not None:
            min_loss = 1e10
            patience = 0
            cfg_patc = self.cfg['patience']
            assert val_name in val_loaders.keys(), "Not specify the dataloader to perform early stopping."
            print("[INFO] training {} epochs, with early stopping on {}, patience is {}.".format(epochs, val_name, cfg_patc))
        else:
            print("[INFO] training {} epochs, without early stopping.".format(epochs))
        
        last_epoch = -1
        for epoch in range(epochs):
            last_epoch = epoch
            steplr_monitor_loss = None

            train_cltor, batch_loss = self._train_each_epoch(train_loader)
            self.writer.add_scalar('loss/train_batch_loss', batch_loss, epoch+1)
            
            if measure:
                train_ci, train_loss = eval_sdata(train_cltor['y_hat'], train_cltor['y'], loss=self.cfg['loss_type'])
                steplr_monitor_loss = train_loss
                self.writer.add_scalar('loss/train_overall_loss', train_loss, epoch+1)
                self.writer.add_scalar('c_index/train_ci', train_ci, epoch+1)
                print('[EVAL] training epoch {}, avg. batch loss: {:.8f}, loss: {:.8f}, c_index: {:.5f}'.format(epoch+1, batch_loss, train_loss, train_ci))

            val_loss = None
            if val_loaders is not None:
                for k in val_loaders.keys():
                    val_cltor = self.test_model(self.model, val_loaders[k], device=self.cfg['device'])
                    met_ci, met_loss = eval_sdata(val_cltor['y_hat'], val_cltor['y'], loss=self.cfg['loss_type'])
                    self.writer.add_scalar('loss/%s_overall_loss'%k, met_loss, epoch+1)
                    self.writer.add_scalar('c_index/%s_ci'%k, met_ci, epoch+1)
                    print("[EVAL] {} epoch {}, loss: {:.8f}, c_index: {:.5f}".format(k, epoch+1, met_loss, met_ci))

                    if k == val_name:
                        val_loss = met_loss
            
            if val_loss is not None:
                # lr scheduler
                steplr_monitor_loss = val_loss

                # best model save
                if val_loss < min_loss:
                    if save:
                        torch.save(self.model.state_dict(), self.best_ckpt_path)
                        print("[INFO] best model saved at epoch {}".format(epoch+1))
                    min_loss = val_loss
                    patience = 0
                else:
                    patience += 1
                
                # early stopping if set 'patience' in config file
                if cfg_patc is not None and patience > cfg_patc:
                    break
            
            if steplr_monitor_loss is not None:
                self.steplr.step(steplr_monitor_loss)

            self.writer.flush()

        if save:
            torch.save(self.model.state_dict(), self.last_ckpt_path)
            print("[INFO] last model saved at epoch {}".format(last_epoch+1))

    def _train_each_epoch(self, train_loader):
        bp_every_iters = self.cfg['bp_every_iters']
        collector = {'y_hat': None, 'y': None}
        bp_collector = {'y_hat': None, 'y': None}
        graph_collector = []
        all_loss  = []

        self.model.train()
        for i, data in enumerate(train_loader):
            # 1. forward propagation
            data = data.to(self.cfg['device'])
            if self.cfg['task'] == 'GraphLSurv':
                cur_out_graphs = {'x':[], 'adj':[], 'mask':[]}
                y_hat = self.model(data, out_anchor_graphs=cur_out_graphs)
                graph_collector.append(cur_out_graphs)
            elif self.cfg['task'] == 'GraphPredict':
                y_hat = self.model(data)
            elif self.cfg['task'] == 'GraphBasic':
                y_hat = self.model(data)

            # y_hat.shape = (B, 1), y.shape = (B,)
            collector = collect_tensor(collector, y_hat.detach().cpu(), data.y.detach().cpu())
            bp_collector = collect_tensor(bp_collector, y_hat, data.y)

            if bp_collector['y'].size(0) % bp_every_iters == 0:
                # 2. backward propagation
                if torch.sum(bp_collector['y'] > 0).item() <= 0:
                    print("[WARNING] batch {}, event count <= 0, skipped.".format(i+1))
                    bp_collector = {'y_hat': None, 'y': None}
                    graph_collector = []
                    continue

                # 2.1 zero gradients buffer
                self.optimizer.zero_grad()

                # 2.2 calculate loss
                loss = self.loss(bp_collector['y_hat'], bp_collector['y'])
                loss += self.loss_reg(self.model.parameters())
                if self.loss_graph is not None:
                    loss1 = .0
                    ag_cnt = 0
                    for anchor_graphs in graph_collector:
                        for ag_x, ag_adj, ag_mask in zip(anchor_graphs['x'], anchor_graphs['adj'], anchor_graphs['mask']):
                            loss1 += self.loss_graph(ag_x, ag_adj, node_mask=ag_mask)
                            ag_cnt += 1
                    loss += loss1 / ag_cnt
                all_loss.append(loss.item())
                print("[EVAL] training batch {}, loss: {:.8f}".format(i+1, loss.item()))

                # 2.3 backwards gradients and update networks
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
                
                bp_collector = {'y_hat': None, 'y': None}
                graph_collector = []

        return collector, utils_avg(all_loss)

    @staticmethod
    def _prepare_dataset(patient_ids, cfg):
        """
        cfg with keys of 'dir_graph_node', 'dir_graph_edge', 'path_csv', 'batch_size', 'num_workers'
        """
        print('[INFO] dataset with undirect graph: {}'.format(cfg['undirect_graph']))
        assert 'k_knn_graph' in cfg
        dir_pat_graph = cfg['dir_pat_graph'].format(cfg['k_knn_graph'])
        dataset = ds.PatchGraphDataset(patient_ids, dir_pat_graph,
            cfg['dir_sld_feat'], cfg['path_csv'], if_force_undirect=cfg['undirect_graph'])
        assert cfg['model_in_dim'] == dataset.num_features
        assert cfg['model_out_dim'] == 1 # must be 1 in survival analysis (CoxPH Hypothesis)
        return dataset

    @staticmethod
    def _loss(loss_name):
        assert loss_name in ['sim-breslow', 'breslow', 'rank-breslow'], 'Invalid loss function'
        if loss_name == 'sim-breslow':
            loss_func = nlog_partial_likelihood
        elif loss_name == 'breslow':
            loss_func = nlog_breslow_likelihood
        elif loss_name == 'rank-breslow':
            loss_func = rank_breslow_likelihood
        else:
            raise NotImplementedError('Loss function {} not recognized.'.format(loss_name))
        return loss_func

    @staticmethod
    def _loss_reg(coef):
        coef = .0 if coef is None else coef
        def loss_reg_l1(model_params):
            if coef <= 1e-8:
                return 0
            else:
                return coef * sum([torch.abs(W).sum() for W in model_params])
        return loss_reg_l1

    @staticmethod
    def _loss_reg_graph(coef_smooth, coef_degree, coef_sparse):
        loss = GraphReg(coef_smooth, coef_degree, coef_sparse)
        return loss

    @staticmethod
    def _optimizier(opt_name, model, **kws):
        if opt_name == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=kws['lr']
            )
        else:
            raise NotImplementedError('Optimizer {} not recognized.'.format(opt_name))

        return optimizer

    @staticmethod
    def test_model(model, loader, device='cpu', checkpoint=None):
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint))

        model.eval()
        res = {'y_hat': None, 'y': None}
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                y_hat = model(data)
                res = collect_tensor(res, y_hat.detach().cpu(), data.y.detach().cpu())
        return res
