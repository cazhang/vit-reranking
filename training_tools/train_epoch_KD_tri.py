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
from datasets.msls import MSLS_embed_tri
from matplotlib import pyplot as plt
plt.switch_backend('agg')

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
	epoch_triplet_loss = 0
	epoch_distill_loss = 0
	startIter = 1  # keep track of batch iter across subsets for logging

	nBatches = (len(train_dataset.qIdx) + int(opt.bs) - 1) // int(opt.bs)

	for subIter in trange(train_dataset.nCacheSubset, desc='Query Subset'):
		
		
		tqdm.write(f'====> Building Cache for subset {subIter} / {train_dataset.nCacheSubset}')
		
		if opt.rand_distill: # random triplet sampling
			train_dataset.update_subcache(None, opt.embed_dim, train_dataset.transform)
		else:
			if opt.distill_hardneg: # use teacher offline hard mining
				train_dataset.update_subcache(teacher_model, opt.num_pcs, train_dataset.teacher_transform)
			else: # use student online hard mining
				train_dataset.update_subcache(model, opt.embed_dim, train_dataset.transform)
			
		training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.kernels,
										  batch_size=opt.bs, shuffle=True,
										  collate_fn=MSLS_embed_tri.collate_fn, pin_memory=cuda)

		tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
		tqdm.write('Cached:	' + humanbytes(torch.cuda.memory_cached()))

		model.train()
		for iteration, (input_s, input_t) in \
				enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(15)), startIter):
			# some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
			# where N = batchSize * (nQuery + nPos + nNeg)
			(query_s, positives_s, negatives_s, negCounts_s, indices_s) = input_s
			(query_t, positives_t, negatives_t, negCounts_t, indices_t) = input_t
			 
			if query_s is None:
				continue  # in case we get an empty batch

			B, C, H, W = query_s.shape
			nNeg = torch.sum(negCounts_s)
			data_input_s = torch.cat([query_s, positives_s, negatives_s])
			data_input_t = torch.cat([query_t, positives_t, negatives_t])
			
			if iteration <= 3:
				save_dir = os.path.join(opt.save_path, 'visual')
				if not os.path.exists(save_dir):
					os.makedirs(save_dir)
				fig, ax = plt.subplots(2, 3, figsize=(10, 5))
				ax = ax.flat
				q_img = input_inv_transform(query_s[0].numpy())
				p_img = input_inv_transform(positives_s[0].numpy())
				n_img = input_inv_transform(negatives_s[0].numpy())
				ax[0].grid(False); ax[0].imshow(q_img)
				ax[1].grid(False); ax[1].imshow(p_img)
				ax[2].grid(False); ax[2].imshow(n_img)
				q_img = input_inv_transform(query_t[0].numpy())
				p_img = input_inv_transform(positives_t[0].numpy())
				n_img = input_inv_transform(negatives_t[0].numpy())
				ax[3].grid(False); ax[3].imshow(q_img)
				ax[4].grid(False); ax[4].imshow(p_img)
				ax[5].grid(False); ax[5].imshow(n_img)
				
				save_name = os.path.join(save_dir, f'{epoch_num}_{iteration}.png')
				fig.savefig(save_name)
				plt.close(fig)
				
			data_input_s = data_input_s.to(device)
			image_encoding, _ = model(data_input_s)
		
			#teacher_model.eval()
			data_input_t = data_input_t.to(device)
			with torch.no_grad():
				target_t, _ = teacher_model(data_input_t)
					
			optimizer.zero_grad()

			# calculate loss for each Query, Positive, Negative triplet
			# due to potential difference in number of negatives have to
			# do it per query, per negative
			loss = 0
			triplet_loss = 0
			distill_loss = 0
			
			if opt.task_loss.lower()=='triplet':
				vladQ, vladP, vladN = torch.split(image_encoding, [B, B, nNeg])
				if opt.tl_weight>1e-6:
					for i, negCount in enumerate(negCounts_s):
						for n in range(negCount):
							negIx = (torch.sum(negCounts_s[:i]) + n).item()
							triplet_loss += criterion(vladQ[i: i + 1], vladP[i: i + 1], vladN[negIx:negIx + 1])

				triplet_loss /= nNeg.float().to(device)  # normalise by actual number of negatives
				del vladQ, vladP, vladN
			elif opt.task_loss.lower()=='supcon':
				labels = torch.cat([torch.arange(B), torch.arange(B+nNeg)]).to(device)
			
				triplet_loss = criterion(image_encoding, labels)
			
			if isinstance(criterion_distill, dict):
				for key, val in criterion_distill.items():
					rkd_w = opt.rkd_distance_weight if 'distance' in key.lower() else opt.rkd_angle_weight
					distill_loss += val(image_encoding, target_t)*rkd_w
			else:
				distill_loss = criterion_distill(image_encoding, target_t) * opt.kd_weight
			
			loss = triplet_loss * opt.tl_weight + distill_loss
			loss.backward()

			# compute model gradiencts
			grads = np.concatenate(
				[p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
			grad_l2, grad_max = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
			LOG.progress_saver['Model Grad'].log('Grad L2', grad_l2, group='L2')
			LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')
			
			triplet_loss_np = triplet_loss.detach().cpu().numpy()
			distill_loss_np = distill_loss.detach().cpu().numpy()
			LOG.progress_saver['Train'].log('Triplet Loss', triplet_loss_np, group='TLoss')
			LOG.progress_saver['Train'].log('Distill Loss', distill_loss_np, group='DLoss')
			

			optimizer.step()
			del data_input_s, data_input_t, image_encoding, target_t

			batch_loss = loss.item()
			epoch_loss += batch_loss
			epoch_triplet_loss += triplet_loss.item()
			epoch_distill_loss += distill_loss.item()

			if iteration % 1000 == 0 or nBatches <= 5:
			
				logging_logger.info(f"==> Epoch[{epoch_num}]({iteration}/{nBatches}): Loss: {batch_loss:.4f}, Triplet Loss: {triplet_loss.item():.4f}, Distill Loss :{distill_loss.item():.4f}")


		startIter += len(training_data_loader)
		del training_data_loader, loss
		optimizer.zero_grad()
		torch.cuda.empty_cache()

	avg_loss = epoch_loss / nBatches
	avg_tloss = epoch_triplet_loss / nBatches
	avg_dloss = epoch_distill_loss / nBatches
	

	logging_logger.info(f"===> Epoch {epoch_num} Complete: Avg. Loss: {avg_loss:.4f}, TLoss:{avg_tloss:.4f}, DLoss: {avg_dloss:.4f}")

	return avg_loss
