from collections import defaultdict, OrderedDict

import torch
import os
import numpy as np
import pickle

def get_files(path, filter):
    for root, dirs, files in os.walk(path):
        for file in files:
            if filter(file):
                yield root, file

# experiment[dir][timestep] = (params, metrics)
data = defaultdict(dict)

for root, file in get_files('.', lambda x: 'rnn' in x):
    ts = int(file.split('_')[0])
    key = os.path.join(*root.split('/')[:-1]) # remove params
    weights = torch.load(os.path.join(root, file))
    data[key][ts] = [OrderedDict((k, v.numpy()) for k, v in weights.items())]

for root, file in get_files('.', lambda x: x == 'data.npz'):
    d = np.load(os.path.join(root, file))
    key = root
    eval_metrics = [(file, d[file].reshape(-1, 50)) for file in d.files]
    ts = (np.arange(eval_metrics[0][1].shape[0])+1)*50

    for i, ts in enumerate(ts.tolist()):
        assert key in data
        try:
            data[key][ts].append(OrderedDict((file, data[i]) for file, data in eval_metrics))
        except KeyError:
            print(f'Skipping {ts}')

for run, timestep in data.items():
    for ts, (weights, metrics) in timestep.items():
        targets = [np.mean(metrics['eval_win_tag']), np.mean(metrics['eval_ep_reward'])]
        print(weights, targets)


