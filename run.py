import torch
import os
import numpy as np

from utils.dilate import Dilate

folder_path = './test_results/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

preds = np.random.randn(100, 100, 10)
trues = np.random.randn(100, 100, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alpha_dilate = 0.5
gamma_dilate = 0.01
batch_size = 1

dilate_metrics, shape_metrics, temporal_metrics = Dilate().dilate_metric(preds, trues, device, alpha_dilate, gamma_dilate, batch_size)

print(len(dilate_metrics), len(dilate_metrics[0]))

np.save(folder_path + 'dilate_metrics.npy', np.array(dilate_metrics))
np.save(folder_path + 'shape_metrics.npy', np.array(shape_metrics))
np.save(folder_path + 'temporal_metrics.npy', np.array(temporal_metrics))