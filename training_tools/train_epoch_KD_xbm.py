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
from training_tools.log_info import log_info 
from training_tools.iter_info import iter_info
from datasets.msls import MSLS_embed_tri, ImagesFromList
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

plt.switch_backend('agg')



def flush_log():
 
    for k in list(log_info.keys()):
        del log_info[k]
		
		
		
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

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

class XBM:
	def __init__(self, opt):
		self.K = opt.xbm_size
		self.dim = opt.embed_dim
		self.feats = torch.zeros(self.K, self.dim).cuda()
		self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
		self.ptr = 0

	@property
	def is_full(self):
		return self.targets[-1].item() != 0

	def get(self):
		if self.is_full:
			return self.feats, self.targets
		else:
			return self.feats[:self.ptr], self.targets[:self.ptr]

	def enqueue_dequeue(self, feats, targets):
		q_size = len(targets)
		if self.ptr + q_size > self.K:
			self.feats[-q_size:] = feats
			self.targets[-q_size:] = targets
			self.ptr = 0
		else:
			self.feats[self.ptr: self.ptr + q_size] = feats
			self.targets[self.ptr: self.ptr + q_size] = targets
			self.ptr += q_size
			
def train_epoch(train_dataset, model, optimizer, criterion, device, epoch_num, opt, logging_logger, LOG, 
				teacher_model=None, criterion_distill=None, xbm=None, validation_dataset=None):
	if device.type == 'cuda':
		cuda = True
	else:
		cuda = False

	epoch_loss = 0
	epoch_triplet_loss = 0
	epoch_distill_loss = 0
	epoch_xbm_loss = 0
	startIter = 1  # keep track of batch iter across subsets for logging

	nBatches = (len(train_dataset.qIdx) + int(opt.bs) - 1) // int(opt.bs)
	
	if epoch_num==0:
		iter_info["iteration"]=0

	for subIter in trange(train_dataset.nCacheSubset, desc='Query Subset'):
		
		
		tqdm.write(f'====> Building Cache for subset {subIter} / {train_dataset.nCacheSubset}')
	
		train_dataset.update_subcache(None, opt.embed_dim, train_dataset.transform)
				
		training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.kernels,
										  batch_size=opt.bs, shuffle=True,
										  collate_fn=MSLS_embed_tri.collate_fn, pin_memory=cuda)

		tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
		tqdm.write('Cached:	' + humanbytes(torch.cuda.memory_cached()))

		model.train()
		
		subIter_acc = epoch_num*train_dataset.nCacheSubset + subIter
		
		for batch_idx, (input_s, input_t) in \
				enumerate(tqdm(training_data_loader, position=2, leave=False, desc='Train Iter'.rjust(15)), startIter):
			# some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
			# where N = batchSize * (nQuery + nPos + nNeg)
		
			iter_info["iteration"] += 1
			#print(iter_info["iteration"])
			(query_s, positives_s, negatives_s, negCounts_s, indices_s, targets_s) = input_s
			(query_t, positives_t, negatives_t, negCounts_t, indices_t, targets_t) = input_t
			#breakpoint()
			 
			if query_s is None:
				continue  # in case we get an empty batch

			B, C, H, W = query_s.shape
			nNeg = torch.sum(negCounts_s)
			if nNeg.numpy()!=opt.bs * train_dataset.nNeg:
				#breakpoint()
				continue
			data_input_s = torch.cat([query_s, positives_s, negatives_s])
			data_input_t = torch.cat([query_t, positives_t, negatives_t])
			
				
			data_input_s = data_input_s.to(device)
			indices_s = torch.tensor(indices_s).to(device)
			image_encoding, _ = model(data_input_s)
			
			if opt.enable_xbm:
				db_list = torch.arange(opt.bs, len(indices_s))
				
				xbm.enqueue_dequeue(image_encoding.detach()[db_list], indices_s[db_list])
			
			
			teacher_model.eval()
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
				
			elif opt.task_loss.lower()=='xbm_triplet' or opt.task_loss.lower()=='ada_xbm_triplet':
				#breakpoint()
				loss = criterion(image_encoding, indices_s, image_encoding, indices_s, train_dataset.qIdx, train_dataset.pIdx, train_dataset.nonNegIdx, train_dataset.qImages, train_dataset.dbImages)
				log_info['batch_loss'] = loss.item()
				log_info["xbm_loss"]=0
				if opt.enable_xbm and epoch_num >= opt.xbm_start_iteration:
					#breakpoint()
					xbm_feats, xbm_targets = xbm.get()
					xbm_loss = criterion(image_encoding, indices_s, xbm_feats, xbm_targets, train_dataset.qIdx,train_dataset.pIdx, train_dataset.nonNegIdx)
					log_info["xbm_loss"] = xbm_loss.item()
					#print(xbm_loss)
					loss = loss + float(opt.xbm_weight) * xbm_loss
				triplet_loss = loss
				if False:
					print('verify loss equivalence...')
					criterion_ref = torch.nn.TripletMarginLoss(margin=opt.margin ** 0.5, p=2, reduction='sum').to(device)
					ref_loss = 0
					vladQ, vladP, vladN = torch.split(image_encoding, [B, B, nNeg])
					if opt.tl_weight>1e-6:
						for i, negCount in enumerate(negCounts_s):
							for n in range(negCount):
								negIx = (torch.sum(negCounts_s[:i]) + n).item()
								ref_loss += criterion_ref(vladQ[i: i + 1], vladP[i: i + 1], vladN[negIx:negIx + 1])

					ref_loss /= nNeg.float().to(device)  # normalise by actual number of negatives
					del vladQ, vladP, vladN
					print(f'ref:{ref_loss:.4f}, my:{triplet_loss:.4f}')
				
			elif opt.task_loss.lower()=='supcon':
				labels = torch.cat([torch.arange(B), torch.arange(B+nNeg)]).to(device)
			
				triplet_loss = criterion(image_encoding, labels)
			
			
			if isinstance(criterion_distill, dict):
				for key, val in criterion_distill.items():
					rkd_w = opt.rkd_distance_weight if 'distance' in key.lower() else opt.rkd_angle_weight
					distill_loss += val(image_encoding, target_t)*rkd_w
			else:
				distill_loss = criterion_distill(image_encoding, target_t) 
		
			triplet_loss *= float(opt.tl_weight)
			distill_loss *= float(opt.kd_weight)
			loss = triplet_loss + distill_loss 
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
			
			#breakpoint()
			if opt.task_loss.lower()=='xbm_triplet' or opt.task_loss.lower()=='ada_xbm_triplet':
				#LOG.progress_saver['Train'].log('Ave Neg', log_info['average_neg'], group='XBM')
				#LOG.progress_saver['Train'].log('Non Zero', log_info['non_zero'], group='XBM')
				LOG.progress_saver['Train'].log('Batch Loss', log_info['batch_loss'], group='XBM')
				LOG.progress_saver['Train'].log('Xbm Loss', log_info['xbm_loss'], group='XBM')
			
			LOG.progress_saver['Train'].log('Triplet Weight', opt.tl_weight, group='Weight')
			LOG.progress_saver['Train'].log('Distill Weight', opt.kd_weight, group='Weight')
			LOG.progress_saver['Train'].log('Xbm Weight', opt.xbm_weight, group='Weight')
			

			optimizer.step()
			del data_input_s, data_input_t, image_encoding, target_t

			epoch_loss += loss.item()
			epoch_triplet_loss += triplet_loss.item()
			epoch_distill_loss += distill_loss.item()
			
			
		# save features on validation
		save_dir = os.path.join(opt.save_path, 'features')
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		model.eval()
		
		val_set_queries = ImagesFromList(validation_dataset.qImages, transform=train_dataset.transform)
		data_loader_queries = DataLoader(dataset=val_set_queries,
										  num_workers=opt.kernels,
										  batch_size=opt.cachebatchsize,
										  shuffle=False, pin_memory=cuda)
		qFeat = np.empty((len(val_set_queries), opt.embed_dim), dtype=np.float32)
		with torch.no_grad():				
			for ii, (data_input, indices) in \
					enumerate(tqdm(data_loader_queries, position=0, leave=False, desc='Compute Feat Iter'.rjust(15)), 1):
				data_input = data_input.to(device)
				image_encoding, _ = model(data_input)
				qFeat[indices.detach().numpy(), :] = image_encoding.detach().cpu().numpy()
			npz_path = f"{save_dir}/feat_{subIter_acc}.npz"
			np.savez(npz_path, qFeat=qFeat)
			print(f"FEATS : \t {npz_path}")
			
		# compute drift if feature exist 
		if subIter_acc>0:
			npz_path_prev = f"{save_dir}/feat_{subIter_acc-1}.npz"
			old_qFeat = np.load(npz_path_prev)['qFeat']
			mse = mean_squared_error(qFeat, old_qFeat)
			LOG.progress_saver['Train'].log('Feat Drift', mse, group='Drift')
		
			del old_qFeat
		
		del qFeat
	
		logging_logger.info(f"==> Epoch[{epoch_num}]({subIter}/{train_dataset.nCacheSubset}): Loss: {loss.item():.4f}, Triplet Loss: {triplet_loss.item():.4f}, Distill Loss :{distill_loss.item():.4f}")
		
		startIter += len(training_data_loader)
	
		optimizer.zero_grad()
		torch.cuda.empty_cache()
		flush_log()

	avg_loss = epoch_loss / nBatches
	avg_tloss = epoch_triplet_loss / nBatches
	avg_dloss = epoch_distill_loss / nBatches
	

	logging_logger.info(f"===> Epoch {epoch_num} Complete: Avg. Loss: {avg_loss:.4f}, TLoss:{avg_tloss:.4f}, DLoss: {avg_dloss:.4f}")

	return avg_loss
