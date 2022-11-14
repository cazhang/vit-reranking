import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import os
import numpy as np
from utilities.diml import Sinkhorn, calc_similarity
from evaluation.metrics import get_metrics_rank, get_metrics
from utilities.visual import visual_heatmap
os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'

def get_resnet_block_output(model, x, final_only=True, layer_id=None):
	
	if final_only:
		out = model(input_img.cuda())
		if isinstance(out, tuple): out, aux_f = out
		return out, aux_f
	else:
		if layer_id is None: layer_id = 1
		x = model.model.maxpool(model.model.relu(model.model.bn1(model.model.conv1(x))))
		for bid, block in enumerate(model.layer_blocks, 1):
			if bid > layer_id: break
			x = block(x)
		b,d,h,w = x.size()
		x = x.view(b,d,-1)
		x = x.permute(0,2,1)# b n d 
	
		return x
		
def evaluate_patch_similarity(model, dataset, dataloader, layer_id):
	model.eval()
	final_iter = tqdm(dataloader, desc='Embedding Data...')
	sims = []
	for idx, inp in enumerate(final_iter):
		input_img, target = inp[1], inp[0]
		patches = get_resnet_block_output(model, input_img.cuda(), final_only=False, layer_id=layer_id)
		b,n,d = patches.size()
		patches = F.normalize(patches, p=2, dim=-1)
		sim = torch.einsum('bmd,bnd->bmn', patches, patches)  # b m n 
		diag_ids = torch.eye(n).expand(b,-1,-1).cuda()
		sim = sim - diag_ids
		avg_sim = torch.sum(sim, dim=(1,2)) / (n*(n-1))# b
		sims.append(avg_sim.cpu().detach())
	sims = torch.cat(sims)
	nimg = sims.size(0)
	sims = torch.mean(sims, 0)
	print(f'evaluated on {nimg} images, and the similarity of layer {layer_id}is {sims.cpu().detach().numpy()}')
	return sims

def evaluate(model, dataset, dataloader, no_training=True, trunc_nums=None, use_uniform=False, grid_size=4, use_inverse=False, temperature=0.1, use_cls_token=False, to_submit=False,plot_topk=False):

	print(f'no_training, {no_training}')

	device = torch.device('cuda')
	model.eval()
	has_head = False
	use_pre = False

	for name, m in model.named_modules():
		if 'head' in name:
			has_head = True
			break

	save_dir = f'/home/czhang/DIML/visual/{model.pars.dataset}_{model.pars.arch}_g{grid_size}'

	print(f'saving to {save_dir}')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	feat_name = os.path.join(save_dir, 'feat.pt')
	hit_list_name = os.path.join(save_dir, 'hit_list.npy')
	if os.path.exists(hit_list_name):
		hit_list = np.load(hit_list_name)
		to_save_hit = False
		print(f'res50 hit list loaded from: {hit_list_name}')
	else:
		hit_list = []
		to_save_hit = True

	with torch.no_grad():
		if os.path.exists(feat_name) and False:
			feats = torch.load(feat_name)
			feature_bank = feats['feature_bank']
			feature_bank_center = feats['feature_bank_center']
			labels = feats['labels']
			feature_bank = feature_bank.to(device)
			feature_bank_center = feature_bank_center.to(device)
			N, C = feature_bank.size(0), feature_bank.size(1)
		else:
			if no_training:
				if (7 % grid_size == 0) or has_head:
					resize = nn.AdaptiveAvgPool2d(grid_size)
				else:
					resize = nn.Sequential(
						nn.Upsample(grid_size * 4, mode='bilinear', align_corners=True),
						nn.AdaptiveAvgPool2d(grid_size),
					)
				feature_bank_center = []

			target_labels = []
			feature_bank = []
			labels = []
			final_iter = tqdm(dataloader, desc='Embedding Data...')

			for idx, inp in enumerate(final_iter):
				#if idx>100: break
				input_img, target = inp[1], inp[0]
				target_labels.extend(target.numpy().tolist())
				out = model(input_img.cuda())
				if isinstance(out, tuple):
					out, aux_f = out

				if no_training:
					enc_out, no_avg_feat = aux_f
					if not use_pre:
						if has_head:
							no_avg_feat = model.model.head(no_avg_feat)
							no_avg_feat = no_avg_feat.permute(0, 2, 1)  # bs x C x L
							no_avg_feat = no_avg_feat.view(no_avg_feat.size(0), -1, int(no_avg_feat.size(-1) ** 0.5), int(no_avg_feat.size(-1) ** 0.5))
						else:
							no_avg_feat = no_avg_feat.transpose(1, 3)
							no_avg_feat = model.model.last_linear(no_avg_feat)
							no_avg_feat = no_avg_feat.transpose(1, 3)
					else:
						out = enc_out

					if no_avg_feat.size(-1) > grid_size:
						no_avg_feat = resize(no_avg_feat)

					feature_bank.append(no_avg_feat.data)
					feature_bank_center.append(out.data)
				else:
					feature_bank.append(out.data)

				labels.append(target)

			feature_bank = torch.cat(feature_bank, dim=0)


			labels = torch.cat(labels, dim=0)
			N, C, H, W = feature_bank.size()
			feature_bank = feature_bank.view(N, C, -1)

			if no_training:
				feature_bank_center = torch.cat(feature_bank_center, dim=0)
			else:
				feature_bank_center = feature_bank.mean(2)

			feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1)
			feature_bank_center = torch.nn.functional.normalize(feature_bank_center, p=2, dim=1)

			# save feature to pt file
			feats = {'feature_bank': feature_bank, 'feature_bank_center': feature_bank_center, 'labels': labels}
			torch.save(feats, feat_name)


		trunc_nums = trunc_nums or [0, 5, 10, 50, 100, 500, 1000]

		overall_r1 = {k: 0.0 for k in trunc_nums}
		overall_rp = {k: 0.0 for k in trunc_nums}
		overall_mapr = {k: 0.0 for k in trunc_nums}

	  	# evaluation starts
		for idx in trange(len(feature_bank)):
			# if idx>10: break
			anchor_center = feature_bank_center[idx]
			approx_sim, uv = calc_similarity(None, anchor_center, None, feature_bank_center, 0)
			approx_sim[idx] = -100

			approx_tops = torch.argsort(approx_sim, descending=True)

			if max(trunc_nums) > 0:
				top_inds = approx_tops[:max(trunc_nums)]

				anchor = feature_bank[idx]
				sim, uv = calc_similarity(anchor, anchor_center, feature_bank[top_inds], feature_bank_center[top_inds], 1, use_uniform, use_inverse, temperature, use_cls_token)

				rank_in_tops = torch.argsort(sim + approx_sim[top_inds], descending=True)

			for trunc_num in trunc_nums:
				if trunc_num == 0:
					final_tops = approx_tops
				else:
					rank_in_tops_real = top_inds[rank_in_tops][:trunc_num]

					final_tops = torch.cat([rank_in_tops_real, approx_tops[trunc_num:]], dim=0)

				r1, rp, mapr = get_metrics_rank(final_tops.data.cpu(), labels[idx], labels)

				if trunc_num>0 and r1>0 and to_save_hit:
					hit_list.append(idx)

				overall_r1[trunc_num] += r1
				overall_rp[trunc_num] += rp
				overall_mapr[trunc_num] += mapr

			top_id = final_tops.data.cpu()[0]
			top_label = labels[top_id]
			query_label = labels[idx]
			top_rank_id = 0

			if plot_topk>1:
				top_id = final_tops.data.cpu()[:plot_topk]
				top_label = labels[top_id]

			if idx<1000 and idx%10==0:
				visual_heatmap(dataset, idx, top_id, query_label, top_label,
							   top_rank_id, uv, save_dir=save_dir,
							   temperature=temperature,
							   use_cls_token=use_cls_token,
							   to_submit=to_submit)


		if to_save_hit:
			np.save(hit_list_name, hit_list)
			print(f'hit list saved to {hit_list_name}')

		for trunc_num in trunc_nums:
			overall_r1[trunc_num] /= float(N / 100)
			overall_rp[trunc_num] /= float(N / 100)
			overall_mapr[trunc_num] /= float(N / 100)

			print(f"trunc_num: {trunc_num}, temperature: {temperature:.2f}")
			print('###########')
			print('Now rank-1 acc=%f, RP=%f, MAP@R=%f' % (overall_r1[trunc_num], overall_rp[trunc_num], overall_mapr[trunc_num]))

	data = {
		'r1': [overall_r1[k] for k in trunc_nums],
		'rp': [overall_rp[k] for k in trunc_nums],
		'mapr': [overall_mapr[k] for k in trunc_nums],
	}
	return data
