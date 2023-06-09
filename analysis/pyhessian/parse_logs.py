import os
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

test_name = 'Vga2'
log_dir = 'pyhessian/logs'
entries = collections.defaultdict(list)
diags = collections.defaultdict(list)
for folder in sorted(os.listdir(log_dir)):
    if '{}_'.format(test_name) in folder:
        input_file_path = os.path.join(log_dir, folder, 'test.log')
        inLog = open(input_file_path, 'r')
        lines = inLog.readlines()
        inLog.close()

        for line in lines:
            if '.pt' in line and '\t' in line:
                method, dataset, run = line.split(':')[2].split('\t')
            elif 'Eigen' in line:
                eigenvalue = float(line[line.find("[")+1:line.find("]")])
            elif 'Trace' in line:
                trace = float(line.split(' ')[1])
            elif 'Acc =' in line:
                acc = line.split('= ')[1].split(' ')[0]
            # elif 'Magnitude' in line:
                # weight_mag = float(line.split(': ')[1])
        entries[(method, dataset)].append({'run': run.strip(), 'eigen': eigenvalue, 'trace': trace, 'acc': acc})
        np_file_path = os.path.join(log_dir, folder, 'diag.npy')
        diag = np.load(np_file_path)
        diags[(method, dataset)].append((run.strip(), diag))
for key in entries:
    # Get diag norms first
    diags_list = diags[key]
    norm_matrix = np.full( (len(diags_list)-1, len(diags_list)-1), 0.0)
    for run, diag in diags_list:
        for run2, diag2 in diags_list:
            if 'server' not in run and 'server' not in run2:
                run_num = int(run.split('.pt')[0].split('_')[-1])
                run2_num = int(run2.split('.pt')[0].split('_')[-1])
                norm_matrix[run_num][run2_num] = (np.linalg.norm(diag) - np.linalg.norm(diag2))**2
                # norm_matrix[run_num][run2_num] = np.dot(diag, diag2)/(np.linalg.norm(diag)*np.linalg.norm(diag2))
    # plt.figure()
    # plt.matshow(norm_matrix, vmin=0.0, vmax=60000)
    # plt.colorbar()
    # plt.savefig('{}/{}_{}.png'.format(log_dir, key[0], test_name))
    print('{} {}'.format(key, np.mean(norm_matrix)))


for key in entries:
    df = pd.DataFrame(entries[key])
    df = df.sort_values('run')
    df.to_csv('{}/{}_{}.csv'.format(log_dir, key[0], test_name))
print('DONE')