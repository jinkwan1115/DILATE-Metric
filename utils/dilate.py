import torch
import numpy as np
from . import soft_dtw
from . import path_soft_dtw 

class Dilate(object):
	def __init__(self):
		pass

	def dilate_metric(self, outputs, targets,device, alpha=0.5, gamma=0.01, batch_size=1):
		"""
		outputs, targets: shape (n_indices, seq_length, n_features)
		alpha: between 0 and 1, the weight of shape error and temporal error
		gamma: the gamma(smoothness) for softdtw
		batch_size: size of each batch for GPU computation
		"""

		outputs = torch.tensor(outputs, dtype=torch.float32, device=device) # [n_indices, seq_length, n_features]
		targets = torch.tensor(targets, dtype=torch.float32, device=device) # [n_indices, seq_length, n_features]
		
		num_indices, seq_length, num_vars = outputs.shape
		softdtw_batch = soft_dtw.SoftDTWBatch.apply
		path_dtw = path_soft_dtw.PathDTWBatch.apply

		# Omega: squared penalization for temporal error
		Omega = soft_dtw.pairwise_distances(
			torch.arange(1, seq_length + 1).view(seq_length, 1).to(device)
		).unsqueeze(0).to(device) # [1, seq_length, seq_length]

		# Results to store
		dilate_metrics_all = [] # list of dilate metric for each index and for each variable
		shape_metrics_all = [] # list of shape metric for each index and for each variable
		temporal_metrics_all = [] # list of temporal metric for each index and for each variable
		
		# Calculate dilate metric for each index
		for idx in range(num_vars):

			shape_metrics_var = []
			temporal_metrics_var = []
			dilate_metrics_var = []

			for batch_start in range(0, num_indices, batch_size):
				batch_end = min(batch_start + batch_size, num_indices)
				outputs_batch = outputs[batch_start:batch_end, :, idx].unsqueeze(2).to(device)  # [batch_size, seq_length, 1]
				targets_batch = targets[batch_start:batch_end, :, idx].unsqueeze(2).to(device)  # [batch_size, seq_length, 1]

				# Flatten over batch and variables
				outputs_flat = outputs_batch.permute(1, 0, 2).contiguous().view(seq_length, -1) # [seq_length, batch_size * 1]
				targets_flat = targets_batch.permute(1, 0, 2).contiguous().view(seq_length, -1) # [seq_length, batch_size * 1]

				D = torch.zeros((batch_size, seq_length, seq_length)).to(device)
				for i in range(min(batch_size, num_indices - batch_start)):
					# Pairwise distance matrix for the batch
					D_sample = soft_dtw.pairwise_distances(targets_flat[:,i].unsqueeze(1), outputs_flat[:,i].unsqueeze(1))  # [seq_length, seq_length]
					D[i,:,:] = D_sample
				
				# Compute shape loss (Soft-DTW)
				shape_metric = softdtw_batch(D, gamma)  
				shape_metrics_var.append(shape_metric.tolist())

				# Compute temporal loss (Path-DTW)
				path = path_dtw(D, gamma)  # [batch_size, seq_length, seq_length]
				temporal_metric = (path * Omega).sum(dim=[1, 2]) / (seq_length * seq_length)
				temporal_metrics_var.extend(temporal_metric.tolist())

				dilate_metric = alpha * shape_metric + (1 - alpha) * temporal_metric
				dilate_metrics_var.extend(dilate_metric.tolist())
				
			shape_metrics_all.append(shape_metrics_var)
			#print(len(shape_metrics_all), len(shape_metrics_all[0]))
			temporal_metrics_all.append(temporal_metrics_var)
			#print(len(temporal_metrics_all), len(temporal_metrics_all[0]))
			dilate_metrics_all.append(dilate_metrics_var)
			#print(len(dilate_metrics_all), len(dilate_metrics_all[0]))

		return dilate_metrics_all, shape_metrics_all, temporal_metrics_all