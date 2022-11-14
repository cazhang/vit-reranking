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

Validation of NetVLAD, using the Mapillary Street-level Sequences Dataset.
'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'

# from RKD paper: https://github.com/lenscloth/RKD/blob/master/metric/loss.py

def pdist(e, squared=False, eps=1e-12):
	e_square = e.pow(2).sum(dim=1)
	prod = e @ e.t()
	res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

	if not squared:
		res = res.sqrt()

	res = res.clone()
	res[range(len(e)), range(len(e))] = 0
	return res
	
class RkdDistance(nn.Module):

	def __init__(self, device):
		super().__init__()
		self.criterion = nn.SmoothL1Loss(reduction='mean').to(device)
		
	def forward(self, student, teacher):
		with torch.no_grad():
			t_d = pdist(teacher, squared=False)
			mean_td = t_d[t_d>0].mean()
			t_d = t_d / mean_td

		s_d = pdist(student, squared=False)
		mean_sd = s_d[s_d>0].mean()
		s_d = s_d / mean_sd

		loss = self.criterion(s_d, t_d)
		return loss
		
class RkdAngle(nn.Module):

	def __init__(self, device):
		super().__init__()
		self.criterion = nn.SmoothL1Loss(reduction='mean').to(device)
		
	def forward(self, student, teacher):
		# N x C
		# N x N x C
		with torch.no_grad():
			td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
			norm_td = F.normalize(td, p=2, dim=2)
			t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

		sd = (student.unsqueeze(0) - student.unsqueeze(1))
		norm_sd = F.normalize(sd, p=2, dim=2)
		s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

		loss = self.criterion(s_angle, t_angle)
		return loss
		
		
# Supervised Contrastive loss: https://github.com/HobbitLong/SupContrast/blob/master/losses.py 
class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""
	def __init__(self, temperature=0.07, contrast_mode='all',
				 base_temperature=0.07):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
		self.contrast_mode = contrast_mode
		self.base_temperature = base_temperature

	def forward(self, features, labels=None, mask=None):
		"""Compute loss for model. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf
		Args:
			features: hidden vector of shape [bsz, n_views, ...].
			labels: ground truth of shape [bsz].
			mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
				has the same class as sample i. Can be asymmetric.
		Returns:
			A loss scalar.
		"""
		device = (torch.device('cuda')
				  if features.is_cuda
				  else torch.device('cpu'))
	
		if len(features.shape) < 3:
			features = features.view(features.shape[0], 1, -1)
		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		batch_size = features.shape[0]
		if labels is not None and mask is not None:
			raise ValueError('Cannot define both `labels` and `mask`')
		elif labels is None and mask is None:
			mask = torch.eye(batch_size, dtype=torch.float32).to(device)
		elif labels is not None:
			labels = labels.contiguous().view(-1, 1)
			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')
			mask = torch.eq(labels, labels.T).float().to(device)
		else:
			mask = mask.float().to(device)

		contrast_count = features.shape[1]
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
		if self.contrast_mode == 'one':
			anchor_feature = features[:, 0]
			anchor_count = 1
		elif self.contrast_mode == 'all':
			anchor_feature = contrast_feature
			anchor_count = contrast_count
		else:
			raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

		# compute logits
		anchor_dot_contrast = torch.div(
			torch.matmul(anchor_feature, contrast_feature.T),
			self.temperature)
		# for numerical stability
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()

		# tile mask
		mask = mask.repeat(anchor_count, contrast_count)
		# mask-out self-contrast cases
		logits_mask = torch.scatter(
			torch.ones_like(mask),
			1,
			torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
			0
		)
		mask = mask * logits_mask

		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

		# compute mean of log-likelihood over positive
		mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-8)

		# loss
		loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
		loss = loss.view(anchor_count, batch_size).mean()

		return loss


#https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py

class HKDLoss(nn.Module):
	"""Hinton KL div loss"""
	def __init__(self, device, opt):
		super(HKDLoss, self).__init__()
		self.T = opt.temperature
		self.bs = opt.bs
		self.criterion = nn.KLDivLoss(reduction='batchmean').to(device)
		
	def forward(self, outputs, teacher_outputs):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha
		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities! See Issue #2
		"""

		input_s, dim_s = outputs.shape
		input_t, dim_t = teacher_outputs.shape 
		assert input_s==input_t
		nNeg = input_s-2*self.bs 
		query_s, pos_s, neg_s = torch.split(outputs, [self.bs, self.bs, nNeg])
		query_t, pos_t, neg_t = torch.split(teacher_outputs, [self.bs, self.bs, nNeg])
		
		try:
			pos_neg_s = torch.cat([pos_s.view(self.bs, 1, dim_s), neg_s.view(self.bs,-1,dim_s)], dim=1)
			pos_neg_t = torch.cat([pos_t.view(self.bs, 1, dim_t), neg_t.view(self.bs,-1,dim_t)], dim=1)
		except:
			breakpoint()
		prob_s = torch.bmm(query_s.view(self.bs, 1, dim_s), pos_neg_s.transpose(1,2)).squeeze(1)
		prob_t = torch.bmm(query_t.view(self.bs, 1, dim_t), pos_neg_t.transpose(1,2)).squeeze(1)
		
		
		KD_loss = self.criterion(F.log_softmax(prob_s/self.T, dim=1),
								 F.softmax(prob_t/self.T, dim=1)) * (self.T * self.T)
		#print(KD_loss)
		return KD_loss
		
class XbmTripletLoss(nn.Module):
	def __init__(self, margin=0.1, opt=None):
		super(XbmTripletLoss, self).__init__()
		self.margin = margin
		self.opt = opt

	def forward(self, inputs_col, targets_col, inputs_row, targets_row, qidxs, pidxs, nnegs, qImages=None, dbImages=None):
		
		n = inputs_col.size(0)
		# Compute similarity matrix
		sim_mat = torch.matmul(inputs_col, inputs_row.t())
		# split the positive and negative pairs
		log_info={}
		loss = 0
		neg_count = 0
		triplet_len = self.opt.nNeg+2
		
		query_list = torch.arange(0, len(inputs_col), triplet_len)
		query_idx = targets_col[query_list]
		qidxs = torch.tensor(qidxs).to(targets_row)

		db_list = torch.tensor([i for i in torch.arange(n) if i not in query_list])
		
		from_batch = torch.equal(targets_col, targets_row)
		targets_row_copy = targets_row.detach().clone()
			
		for i in range(len(query_list)):
			#breakpoint()
			idx = query_list[i]
			qidx = query_idx[i]
			if qidx in qidxs:
				qidx_local = (qidxs==qidx).nonzero()
			else:
				print('not found query')
				continue 
			
			this_pidx = torch.tensor(pidxs[qidx_local]).to(targets_row)
			this_nneg = torch.tensor(nnegs[qidx_local]).to(targets_row)
			# remove query entries
			if from_batch:
				targets_row_copy[query_list]=-1
			
			pos_pair_idx = torch.nonzero(torch.isin(targets_row, this_pidx)).view(-1)
			if pos_pair_idx.shape[0] > 0:
				pos_pair_ = sim_mat[idx, pos_pair_idx]
				pos_pair_ = torch.sort(pos_pair_)[0]

				neg_pair_idx = torch.isin(targets_row_copy, this_nneg, invert=True)
				if from_batch:
					neg_pair_idx[query_list]=False 
				neg_pair_idx = torch.nonzero(neg_pair_idx).view(-1)
				neg_pair_ = sim_mat[idx, neg_pair_idx]
				neg_pair_ = torch.sort(neg_pair_)[0]

				select_pos_pair_idx = torch.nonzero(
					pos_pair_ < neg_pair_[-1] + self.margin
				).view(-1)
				pos_pair = pos_pair_[select_pos_pair_idx]

				select_neg_pair_idx = torch.nonzero(
					neg_pair_ > max(0.4, pos_pair_[-1]) - self.margin
				).view(-1)
				neg_pair = neg_pair_[select_neg_pair_idx]
				#breakpoint()
				if len(pos_pair) > 0:
					pos_loss = torch.sum(1 - pos_pair) / len(pos_pair)
				else:
					pos_loss = 0
				if len(neg_pair) > 0:
					neg_count += len(neg_pair)
					neg_loss = torch.sum(neg_pair) / len(neg_pair)
				else:
					neg_loss = 0
				loss+=(pos_loss + neg_loss)
			
			
			if i==0 and False:
				print(f'pos:{len(pos_pair)}, neg:{len(neg_pair)}, loss:{loss:.3f}')
		loss = loss / torch.FloatTensor([len(query_list)]).to(targets_row)
		

		return loss
		
# modify xbm triplet to support sum, and mean reduction
class AdaXbmTripletLoss(nn.Module):
	def __init__(self, margin=0.1, reduction='mean', opt=None):
		super(AdaXbmTripletLoss, self).__init__()
		self.margin = margin
		self.reduction = reduction
		self.opt = opt
		self.criterion = nn.TripletMarginLoss(margin=margin ** 0.5, p=2, reduction='sum').to(opt.device)

	def forward(self, inputs_col, targets_col, inputs_row, targets_row, 
				qidxs, pidxs, nnegs, qImages=None, dbImages=None, same_batch_neg=False):
		
		n = inputs_col.size(0)
		# Compute similarity matrix
		sim_mat = torch.matmul(inputs_col, inputs_row.t())
		# split the positive and negative pairs
		log_info={}
		loss = 0
		neg_count = 0
		
		#breakpoint()
		query_list = torch.arange(self.opt.bs) # 0,1,2
		query_idx = targets_col[query_list] # real query image index 
		qidxs = torch.tensor(qidxs).to(targets_row)
		#query_idx_local =  torch.cat([(qidxs==i).nonzero() for i in query_idx]) # index to qidxs 

		db_list = torch.tensor([i for i in torch.arange(n) if i not in query_list]) # database index: 1,2,3...
			
		from_batch = torch.equal(targets_col, targets_row) 
		targets_row_copy = targets_row.detach().clone()
		
		for i in range(len(query_list)):
	
			idx = query_list[i] # for sim mat index 
			qidx = query_idx[i] # for input loading
			
			if qidx in qidxs:
				qidx_local = (qidxs==qidx).nonzero() # for aux list index 
			else:
				continue
			
			
			this_pidx = torch.tensor(pidxs[qidx_local]).to(targets_row)
			this_nneg = torch.tensor(nnegs[qidx_local]).to(targets_row)
			# remove query entries when both from the same minibatch, otherwise, targets_row all from database
			if from_batch:
				# mask out query
				targets_row_copy[query_list] = -1
			
			#pos_pair_idx = torch.nonzero(torch.isin(targets_row, this_pidx)).view(-1)
			
			pos_pair_idx = self.opt.bs+i
			
			pos_pair_ = sim_mat[idx, pos_pair_idx]
			pos_pair_ = torch.sort(pos_pair_)[0] # 0.5, 0.6, 0.8...
			
			if same_batch_neg:
				neg_pair_idx = torch.arange(self.opt.bs*2+i*(self.opt.nNeg), self.opt.bs*2+(i+1)*self.opt.nNeg)
			else:
				neg_pair_idx = torch.isin(targets_row_copy, this_nneg, invert=True)
				if from_batch:
					neg_pair_idx[query_list]=False	
				neg_pair_idx = torch.nonzero(neg_pair_idx).view(-1)
			
			neg_pair_ = sim_mat[idx, neg_pair_idx]
			#neg_pair_, neg_ind = torch.sort(neg_pair_) # 0.2, 0.3, 0.6...

			#select_pos_pair_idx = torch.nonzero(pos_pair_ < neg_pair_[-1] + self.margin).view(-1) # 0.5, 0.6 
			pos_pair = pos_pair_

			select_neg_pair_idx = torch.nonzero(
				neg_pair_ > pos_pair - self.margin
			).view(-1) # 0.6
			neg_pair = neg_pair_[select_neg_pair_idx]
			neg_pair_idx = neg_pair_idx[select_neg_pair_idx]
			
			
			if len(neg_pair) > 0:
				neg_count+=len(neg_pair)
				for n in neg_pair_idx:
					loss+=self.criterion(inputs_col[idx,:], inputs_row[pos_pair_idx,:], inputs_row[n,:])
	
		del targets_row_copy
		
		if neg_count>0:
			#print(neg_count)
			loss = loss / torch.tensor(neg_count).float().to(targets_row)
		else:
			loss = torch.tensor(loss).float().to(targets_row)
		#breakpoint()
		return loss
