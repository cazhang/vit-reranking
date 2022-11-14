'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Significant parts of our code are based on [Nanne's pytorch-netvlad repository]
(https://github.com/Nanne/pytorch-NetVlad/), as well as some parts from the [Mapillary SLS repository]
(https://github.com/mapillary/mapillary_sls)

Trains an epoch of NetVLAD, using the Mapillary Street-level Sequences Dataset.
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
				teacher_model=None, criterion_distill=None):
	if device.type == 'cuda':
		cuda = True
	else:
		cuda = False

	epoch_loss = 0
	startIter = 1  # keep track of batch iter across subsets for logging

	nBatches = (len(train_dataset.qIdx) + int(opt.bs) - 1) // int(opt.bs)

	for subIter in trange(train_dataset.nCacheSubset, desc='Query Subset'):
		
		if False:
			if opt.distill:
				if epoch_num==0: # only do once
					tqdm.write(f'====> Building Fixed triplets for subset {subIter} / {train_dataset.nCacheSubset}')
					train_dataset.generate_triplets(teacher_model, opt.embed_dim)
				train_dataset.set_triplets_increase_sub()
			elif opt.rand_distill:
				tqdm.write(f'====> Sampling triplets using precomputed candidates for subset {subIter} / {train_dataset.nCacheSubset}')
				if epoch_num==0:
					train_dataset.generate_postive_negative_candidates(teacher_model, opt.embed_dim)
				train_dataset.sample_triplets_from_candidates()
				train_dataset.increase_subset_index()
			elif opt.rand_triplets:
				tqdm.write(f'====> Random sampling triplets for subset {subIter} / {train_dataset.nCacheSubset}')
				train_dataset.generate_triplets(teacher_model, opt.embed_dim)
				train_dataset.set_triplets_increase_sub()
		else:
			tqdm.write(f'====> Building Cache for subset {subIter} / {train_dataset.nCacheSubset}')
			train_dataset.update_subcache(model, opt.embed_dim)
			
		training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.kernels,
										  batch_size=opt.bs, shuffle=True,
										  collate_fn=MSLS.collate_fn, pin_memory=cuda)

		tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
		tqdm.write('Cached:	' + humanbytes(torch.cuda.memory_cached()))

		model.train()
		for iteration, (query, positives, negatives, negCounts, indices) in \
				enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(15)), startIter):
			# some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
			# where N = batchSize * (nQuery + nPos + nNeg)
			if query is None:
				continue  # in case we get an empty batch

			B, C, H, W = query.shape
			nNeg = torch.sum(negCounts)
			data_input = torch.cat([query, positives, negatives])

			if iteration <= 3:
				save_dir = os.path.join(opt.save_path, 'visual')
				if not os.path.exists(save_dir):
					os.makedirs(save_dir)
				fig, ax = plt.subplots(1, 3, figsize=(10, 5))
				ax = ax.flat
				q_img = input_inv_transform(query[0].numpy())
				p_img = input_inv_transform(positives[0].numpy())
				n_img = input_inv_transform(negatives[0].numpy())
				ax[0].grid(False)
				ax[0].imshow(q_img)
				ax[1].grid(False)
				ax[1].imshow(p_img)
				ax[2].grid(False)
				ax[2].imshow(n_img)
				save_name = os.path.join(save_dir, f'{epoch_num}_{iteration}.png')
				fig.savefig(save_name)
				plt.close(fig)
				
			data_input = data_input.to(device)
			image_encoding, _ = model(data_input)
		
			vladQ, vladP, vladN = torch.split(image_encoding, [B, B, nNeg])

			optimizer.zero_grad()

			# calculate loss for each Query, Positive, Negative triplet
			# due to potential difference in number of negatives have to
			# do it per query, per negative
			loss = 0
			for i, negCount in enumerate(negCounts):
				for n in range(negCount):
					negIx = (torch.sum(negCounts[:i]) + n).item()
					loss += criterion(vladQ[i: i + 1], vladP[i: i + 1], vladN[negIx:negIx + 1])

			loss /= nNeg.float().to(device)  # normalise by actual number of negatives
			loss.backward()

			# compute model gradiencts
			grads = np.concatenate(
				[p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
			grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
			LOG.progress_saver['Model Grad'].log('Grad L2', grad_l2, group='L2')
			LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

			optimizer.step()
			del data_input, image_encoding, vladQ, vladP, vladN
			del query, positives, negatives

			batch_loss = loss.item()
			epoch_loss += batch_loss

			if iteration % 1000 == 0 or nBatches <= 5:
			
				logging_logger.info(f"==> Epoch[{epoch_num}]({iteration}/{nBatches}): Loss: {batch_loss:.4f}")

				tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
				tqdm.write('Cached:	' + humanbytes(torch.cuda.memory_cached()))

		startIter += len(training_data_loader)
		del training_data_loader, loss
		optimizer.zero_grad()
		torch.cuda.empty_cache()

	avg_loss = epoch_loss / nBatches

	logging_logger.info(f"===> Epoch {epoch_num} Complete: Avg. Loss: {avg_loss:.4f}")

	return avg_loss
