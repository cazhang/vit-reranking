'''

Trains an epoch of CVT, using the Mapillary Street-level Sequences Dataset.

Using trained NetVLAD as teacher to distill CvT: enforce similar embeddings 
'''

import os
import torch
import numpy as np
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader
from training_tools.tools import humanbytes
from datasets.msls import MSLS
from matplotlib import pyplot as plt

def input_inv_transform(x):
	# x: normalised image (mean, std)
	# return: np.uint8, without mean and std
	assert x.ndim==3 and x.shape[0]==3
	std = [0.229, 0.224, 0.225]
	mean = [0.485, 0.456, 0.406]
	x[0, :, :] = x[0, :, :] * std[0] + mean[0]
	x[1, :, :] = x[1, :, :] * std[1] + mean[1]
	x[2, :, :] = x[2, :, :] * std[2] + mean[2]
	x = np.uint8(x*255)
	x = np.transpose(x, (1,2,0))
	return x

def train_epoch(train_dataset, model, optimizer, criterion, device, epoch_num, opt, logging_logger, LOG, 
				teacher_model):
	if device.type == 'cuda':
		cuda = True
	else:
		cuda = False

	epoch_loss = 0
	startIter = 1  # keep track of batch iter across subsets for logging

	nBatches = (len(train_dataset.images) + int(opt.bs) - 1) // int(opt.bs)

	training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.kernels,
										  batch_size=opt.bs, shuffle=True, pin_memory=cuda)
										  
	

	model.train()
	for iteration, (input_s, input_t, input_names) in enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(15)), startIter):
		# some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
		# where N = batchSize * (nQuery + nPos + nNeg)

		B, C, H, W = input_s.shape
		
		if iteration <= 3:
			save_dir = os.path.join(opt.save_path, 'visual')
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)
			fig, ax = plt.subplots(1, 2, figsize=(10, 5))
			ax = ax.flat
			img_s = input_inv_transform(input_s[0].numpy())
			img_t = input_inv_transform(input_t[0].numpy())
		
			ax[0].grid(False)
			ax[0].imshow(img_s)
			ax[1].grid(False)
			ax[1].imshow(img_t)
		
			save_name = os.path.join(save_dir, f'{epoch_num}_{iteration}.png')
			fig.savefig(save_name)
			plt.close(fig)
			
		input_s = input_s.to(device)
		encoding_s, _ = model(input_s)
		input_t = input_t.to(device)
		teacher_model.eval()
		with torch.no_grad():
			target_t, _ = teacher_model(input_t)

		optimizer.zero_grad()

		# calculate loss for each Query, Positive, Negative triplet
		# due to potential difference in number of negatives have to
		# do it per query, per negative
		loss = 0
		loss += criterion(encoding_s, target_t)
		loss.backward()

		# compute model gradiencts
		grads = np.concatenate(
			[p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
		grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
		LOG.progress_saver['Model Grad'].log('Grad L2', grad_l2, group='L2')
		LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

		optimizer.step()

		batch_loss = loss.item()
		epoch_loss += batch_loss

		if iteration % 1000 == 0 or nBatches <= 5:
		
			logging_logger.info(f"==> Epoch[{epoch_num}]({iteration}/{nBatches}): Loss: {batch_loss:.4f}")

			tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
			tqdm.write('Cached:	' + humanbytes(torch.cuda.memory_cached()))

	del training_data_loader, loss
	optimizer.zero_grad()
	torch.cuda.empty_cache()

	avg_loss = epoch_loss / nBatches

	logging_logger.info(f"===> Epoch {epoch_num} Complete: Avg. Loss: {avg_loss:.4f}")

	return avg_loss
