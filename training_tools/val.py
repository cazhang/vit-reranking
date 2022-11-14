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
import numpy as np
import torch
import faiss
import cv2
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets.msls import ImagesFromList, input_transform
from utilities.visual import visual_heatmap_msls
import torch.nn as nn
os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'

import matplotlib.pyplot as plt
from utilities.diml import calc_similarity

def visualise_ret(qid, preds, query_dataset, database_dataset, save_dir, epoch, topk=3, save_str='query'):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	q_name = query_dataset.images[qid]
	names = []; names.append(q_name)
	for i in range(topk):
		db_id = preds[i]
		db_name = database_dataset.images[db_id]
		names.append(db_name)

	n_imgs = len(names)
	fig, axs = plt.subplots(1, n_imgs, figsize=(14, 8))
	axs = axs.flat
	for i in range(n_imgs):
		img = cv2.imread(names[i])[:,:,::-1]
		axs[i].imshow(img)
		axs[i].set_axis_off()
	save_name = os.path.join(save_dir, f'{save_str}{qid:04}_ep{epoch:04}.png')
	fig.savefig(save_name, bbox_inches='tight')
	plt.close(fig)


def val(eval_set, model, device, opt, writer, epoch_num=0, is_train=False, pbar_position=0, trunc_nums=None, grid_size=7):
	if device.type == 'cuda':
		cuda = True
	else:
		cuda = False
	trunc_nums = trunc_nums or [0, 100]
	eval_set_queries = ImagesFromList(eval_set.qImages, transform=input_transform(resize=(opt.imageresizeh, opt.imageresizew)))
	eval_set_dbs = ImagesFromList(eval_set.dbImages, transform=input_transform(resize=(opt.imageresizeh, opt.imageresizew)))
	test_data_loader_queries = DataLoader(dataset=eval_set_queries,
										  num_workers=opt.kernels,
										  batch_size=opt.cachebatchsize,
										  shuffle=False, pin_memory=cuda)
	test_data_loader_dbs = DataLoader(dataset=eval_set_dbs,
									  num_workers=opt.kernels,
									  batch_size=opt.cachebatchsize,
									  shuffle=False, pin_memory=cuda)
	if (7 % grid_size == 0):
		resize = nn.AdaptiveAvgPool2d(grid_size)

	save_dir = os.path.join(opt.save_path, 'visual')

	model.eval()
	with torch.no_grad():
		tqdm.write('====> Extracting Features')

		qFeat = np.empty((len(eval_set_queries), opt.embed_dim), dtype=np.float32)
		dbFeat = np.empty((len(eval_set_dbs), opt.embed_dim), dtype=np.float32)
		qFeat_dense = []
		dbFeat_dense = []
		for feat, feat_dense, test_data_loader in zip([qFeat, dbFeat], [qFeat_dense, dbFeat_dense], [test_data_loader_queries, test_data_loader_dbs]):
			for iteration, (data_input, indices) in \
					enumerate(tqdm(test_data_loader, position=pbar_position, leave=False, desc='Test Iter'.rjust(15)), 1):

				data_input = data_input.to(device)
			
				
				image_encoding, aux_feat = model(data_input)
				if isinstance(aux_feat, tuple) and aux_feat[1] is not None and max(trunc_nums)>0:
					out, no_avg_feat = aux_feat
					if opt.is_parallel:
						no_avg_feat = model.module.model.head(no_avg_feat)
					else:
						no_avg_feat = model.model.head(no_avg_feat)
					no_avg_feat = no_avg_feat.permute(0, 2, 1)  # bs x C x L
					no_avg_feat = no_avg_feat.view(no_avg_feat.size(0), -1, int(no_avg_feat.size(-1) ** 0.5),
												   int(no_avg_feat.size(-1) ** 0.5))
					if no_avg_feat.size(-1) != grid_size:
						no_avg_feat = resize(no_avg_feat)
					no_avg_feat = no_avg_feat.view(no_avg_feat.size(0), no_avg_feat.size(1), -1)  # bs x C x L

					feat_dense.append(no_avg_feat.data.cpu().detach())

				feat[indices.detach().numpy(), :] = image_encoding.detach().cpu().numpy()

				del data_input, image_encoding

	del test_data_loader_queries, test_data_loader_dbs

	if len(qFeat_dense)>0 and len(dbFeat_dense)>0:
		qFeat_dense = torch.cat(qFeat_dense, dim=0)
		dbFeat_dense = torch.cat(dbFeat_dense, dim=0)
		qFeat_dense = torch.nn.functional.normalize(qFeat_dense, p=2, dim=1)
		dbFeat_dense = torch.nn.functional.normalize(dbFeat_dense, p=2, dim=1)
	else:
		trunc_nums = [0]

	tqdm.write('====> Building faiss index')
	faiss_index = faiss.IndexFlatL2(opt.embed_dim)
	# noinspection PyArgumentList
	faiss_index.add(dbFeat)

	tqdm.write('====> Calculating recall @ N')
	n_values = [1, 5, 10, 20, 50, 100]
	
	# for each query get those within threshold distance
	gt = eval_set.all_pos_indices
	final_tops = []
	final_preds = None
	# any combination of mapillary cities will work as a val set
	qEndPosTot = 0
	dbEndPosTot = 0
	for cityNum, (qEndPos, dbEndPos) in enumerate(zip(eval_set.qEndPosList, eval_set.dbEndPosList)):
		faiss_index = faiss.IndexFlatL2(opt.embed_dim)
		dbFeat_global_city = dbFeat[dbEndPosTot:dbEndPosTot+dbEndPos, :]
		qFeat_global_city = qFeat[qEndPosTot:qEndPosTot+qEndPos, :]
		# np solution
		#faiss_index.add(dbFeat_global_city)
		#dists, preds_ = faiss_index.search(qFeat_global_city, max(n_values))
		qFeat_global_city = torch.tensor(qFeat_global_city).to(device)
		dbFeat_global_city = torch.tensor(dbFeat_global_city).to(device)
		# tensor solution
		approx_sim = torch.einsum('nc,mc->nm', qFeat_global_city, dbFeat_global_city)
		preds = torch.argsort(approx_sim, descending=True)[:,:max(n_values)]
		print(preds.shape)
		if max(trunc_nums)>0:
			top_inds = preds[:, :max(trunc_nums)]
			qFeat_dense_city = qFeat_dense[qEndPosTot:qEndPosTot+qEndPos, :]
			dbFeat_dense_city = dbFeat_dense[dbEndPosTot:dbEndPosTot+dbEndPos, :]
			final_tops = []
			for idx in range(len(qFeat_dense_city)):
				top_ind = top_inds[idx]
				anchor_center = qFeat_global_city[idx]
				anchor = qFeat_dense_city[idx].to(device)
				feats_center = dbFeat_global_city[top_ind]
				feats = dbFeat_dense_city[top_ind].to(device)

				
				sim, uv = calc_similarity(anchor, anchor_center, feats, feats_center,
										  stage=1,
										  use_uniform=False,
										  use_inverse=False,
										  temperature=0.1,
										  use_cls_token=True,
										  ot_temp=0.05,
										  use_minus=True,
										  ot_part=1.0)

				rank_in_tops = torch.argsort(sim + approx_sim[idx, top_ind], descending=True)
				rank_in_tops_real = top_ind[rank_in_tops][:max(trunc_nums)]
				final_top = torch.cat([rank_in_tops_real, preds[idx][max(trunc_nums):]], dim=0)
				final_tops.append(final_top)
				if idx % opt.vis_freq==0:
					visual_heatmap_msls(eval_set_queries, eval_set_dbs, idx, rank_in_tops_real[0], rank_in_tops[0], uv, save_dir, use_cls_token=True, to_submit=False, city_num=cityNum)

			final_tops = torch.stack(final_tops, dim=0)
			final_tops = final_tops.cpu().detach().numpy()
			
		preds = preds.cpu().detach().numpy()

		if cityNum == 0:
			predictions = preds	
		else:
			predictions = np.vstack((predictions, preds))
			
		if max(trunc_nums)>0:
			if cityNum == 0:
				final_preds = final_tops
			else:
				final_preds = np.vstack((final_preds, final_tops))
		
		qEndPosTot += qEndPos
		dbEndPosTot += dbEndPos

	correct_at_n = np.zeros(len(n_values))
	final_at_n = np.zeros(len(n_values))
	# TODO can we do this on the matrix in one go?
	for qIx, pred in enumerate(predictions):
		# visualise it
		if qIx % opt.save_freq==0:
			visualise_ret(qIx, pred, query_dataset=eval_set_queries,
						  database_dataset=eval_set_dbs,
						  save_dir=save_dir, epoch=epoch_num, save_str='global_trainval' if is_train else 'global')

		for i, n in enumerate(n_values):
			# if in top N then also in top NN, where NN > N
			if np.any(np.in1d(pred[:n], gt[qIx])):
				correct_at_n[i:] += 1
				break
	recall_at_n = correct_at_n / len(eval_set.qIdx)

	if final_preds is not None:
		for qIx, pred in enumerate(final_preds):
			# visualise it
			if qIx % opt.save_freq==0:
				visualise_ret(qIx, pred, query_dataset=eval_set_queries,
							  database_dataset=eval_set_dbs,
							  save_dir=save_dir, epoch=epoch_num, save_str='diml_trainval' if is_train else 'diml')

			for i, n in enumerate(n_values):
				# if in top N then also in top NN, where NN > N
				if np.any(np.in1d(pred[:n], gt[qIx])):
					final_at_n[i:] += 1
					break

		final_at_n = final_at_n / len(eval_set.qIdx)

	all_recalls = {}  # make dict for output
	for i, n in enumerate(n_values):

		all_recalls[f'globalR_{n}'] = recall_at_n[i]
		tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
		if final_preds is not None:
			all_recalls[f'dimlR_{n}'] = final_at_n[i]
			tqdm.write("====> DIML Recall@{}: {:.4f}".format(n, final_at_n[i]))
	return all_recalls
