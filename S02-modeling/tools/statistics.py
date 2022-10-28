import sys
import os.path as osp
import numpy as np
import pandas as pd

################################
# Normal configuration
CFG = {'TCGA_BRCA':{}, 'NLST': {}}
CFG['TCGA_BRCA']['root_dir'] = '/home/liup/repo/GraphLSurv/S02-modeling/results-ablation/TCGA_BRCA'
CFG['TCGA_BRCA']['list_seed'] = [360, 568, 630, 788, 850]
CFG['NLST']['root_dir'] = '/home/liup/repo/GraphLSurv/S02-modeling/results-ablation/NLST'
CFG['NLST']['list_seed'] = [29, 44, 46, 55, 86]
################################

################################
# Hyopt params
ParamsSet = dict()
ParamsSet['dir_name'] = 'GraphLSurv-s3_gr{}-hyopt-seed_split_{}-smoothness_ratio_{}-agl_ratio_anchors_{}-agl_epsilon_{}'
ParamsSet['list_gr'] = [0.0, 0.1, 0.3, 0.5]
ParamsSet['list_anchors_ratio'] = [0.1, 0.2, 0.3]
ParamsSet['list_smooth_ratio'] = [0.1, 0.2, 0.3]
ParamsSet['list_epsilon'] = [0.8, 0.9]
ParamsSet['result_filename'] = 'stats-{}-hyopt.csv'
################################

################################
# Ablation params
ParamsSet1 = dict()
ParamsSet1['dir_name'] = 'Smooth-seed_split_{}-smoothness_ratio_{}'
ParamsSet1['list_params'] = [0.0, 0.2, 0.3, 0.4]
ParamsSet1['result_filename'] = 'Stats-{}-Ablation-Smooth.csv'
################################


def get_5f(a):
    return float(format(a, '.5f'))

def get_info(read_dir):
    read_path = osp.join(read_dir, 'metrics.txt')

    with open(read_path) as f:
        data = f.readlines()

    line = data[1].strip()
    assert line[:11] == 'test/cindex'
    test_ci = float(line[27:])

    return test_ci

def stats_hyopt(root_read_dir, save_path, list_seed):
    records = {
        'KNN_graph_ratio':[], 
        'Anchors_ratio':[],
        'Smooth_ratio':[],
        'Epsilon':[],
        'Test-1':[],
        'Test-2':[],
        'Test-3':[],
        'Test-4':[],
        'Test-5':[],
        'Mean':[],
        'Std.':[]
    }
    for s0 in ParamsSet['list_gr']:
        for s1 in ParamsSet['list_anchors_ratio']:
            for s2 in ParamsSet['list_smooth_ratio']:
                for s3 in ParamsSet['list_epsilon']:
                    records['KNN_graph_ratio'].append(s0)
                    records['Anchors_ratio'].append(s1)
                    records['Smooth_ratio'].append(s2)
                    records['Epsilon'].append(s3)
                    metrics_ci = []
                    for i, seed in enumerate(list_seed):
                        sub_dir_name = ParamsSet['dir_name'].format(s0, seed, s2, s1, s3)
                        dir_name = osp.join(root_read_dir, sub_dir_name)

                        test_ci = get_info(dir_name)
                        test_ci = get_5f(test_ci)

                        records['Test-{}'.format(i+1)].append(test_ci)
                        metrics_ci.append(test_ci)

                    print(metrics_ci)
                    a = np.array(metrics_ci)
                    records['Mean'].append(get_5f(a.mean()))
                    records['Std.'].append(get_5f(a.std()))

    df = pd.DataFrame(
        records, 
        columns=['KNN_graph_ratio', 'Anchors_ratio', 'Smooth_ratio', 'Epsilon',
            'Test-1', 'Test-2', 'Test-3', 'Test-4', 'Test-5', 'Mean', 'Std.']
    )
    df.to_csv(save_path, index=False)

def stats_ablation(root_read_dir, save_path, list_seed):
    records = {
        'Params':[], 
        'Test-1':[],
        'Test-2':[],
        'Test-3':[],
        'Test-4':[],
        'Test-5':[],
        'Mean':[],
        'Std.':[]
    }
    for s0 in ParamsSet1['list_params']:
        records['Params'].append(s0)
        metrics_ci = []
        for i, seed in enumerate(list_seed):
            sub_dir_name = ParamsSet1['dir_name'].format(seed, s0)
            print(sub_dir_name)
            dir_name = osp.join(root_read_dir, sub_dir_name)

            test_ci = get_info(dir_name)
            test_ci = get_5f(test_ci)

            records['Test-{}'.format(i+1)].append(test_ci)
            metrics_ci.append(test_ci)

        print(metrics_ci)
        a = np.array(metrics_ci)
        records['Mean'].append(get_5f(a.mean()))
        records['Std.'].append(get_5f(a.std()))

    df = pd.DataFrame(
        records, 
        columns=['Params', 'Test-1', 'Test-2', 'Test-3', 'Test-4', 'Test-5', 'Mean', 'Std.']
    )
    df.to_csv(save_path, index=False)


def main(data_name, opt):
    assert data_name in ('NLST', 'TCGA_BRCA')
    assert opt in ('hyopt', 'ablation')

    root_dir = CFG[data_name]['root_dir']
    list_seed = CFG[data_name]['list_seed']
    if opt == 'hyopt':
        save_filename = ParamsSet['result_filename'].format(data_name)
        save_path = osp.join(root_dir, save_filename)
        stats_hyopt(root_dir, save_path, list_seed)
    elif opt == 'ablation':
        save_filename = ParamsSet1['result_filename'].format(data_name)
        save_path = osp.join(root_dir, save_filename)
        stats_ablation(root_dir, save_path, list_seed)

# python3 statistics.py NLST hyopt/ablation
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
