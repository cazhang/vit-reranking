import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm, trange
from utilities.diml import Sinkhorn, calc_similarity_cvt, calc_similarity_featvit, calc_similarity, calc_similarity_cvt_rollout
from utilities.visual import visual_heatmap, visual_self_cross_flow, input_inv_transform, visual_patch_sim, visual_attention_rollout_images, visual_attention_rollout_layers, visual_attention_rollout_images_mean
#from utilities.cam import get_cam_ouput, demo
#from pytorch_grad_cam.utils.image import show_cam_on_image
from evaluation.metrics import get_metrics_rank, get_metrics
from einops import rearrange
import numpy as np
import random 

os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'

# Unlike DIML which uses final layer feature for cross correlation, this code use attention map from vit for optimal transport computation

def get_qk(model, x, blk_ind=0):
	input = x

	x, cls_tokens = model.stage0(x)
	x, cls_tokens = model.stage1(x)

	assert cls_tokens is None 
	x = model.stage2.patch_embed(x)
	B, C, H, W = x.size()

	x = rearrange(x, 'b c h w -> b (h w) c')
	cls_tokens = None
	if model.stage2.cls_token is not None:
		# stole cls_tokens impl from Phil Wang, thanks
		cls_tokens = model.stage2.cls_token.expand(B, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
	
	for i, blk in enumerate(model.stage2.blocks):
		#if i < (model.stage2.depth-1):
		if i < blk_ind:
			x = blk(x, H, W)
		else:
			x = blk.norm1(x)
			if (blk.attn.conv_proj_q is not None or blk.attn.conv_proj_k is not None or blk.attn.conv_proj_v is not None):
				q, k, v = blk.attn.forward_conv(x, H, W)

			q = rearrange(blk.attn.proj_q(q), 'b t (h d) -> b h t d', h=blk.attn.num_heads)
			#k = rearrange(blk.attn.proj_k(k), 'b t (h d) -> b h t d', h=blk.attn.num_heads)
			#v = rearrange(blk.attn.proj_v(v), 'b t (h d) -> b h t d', h=blk.attn.num_heads)

			#attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * blk.attn.scale
			return q

def resize_attn_map(attn, resize, stage, grid, blk_id):
	# attn: b x t x t
	new_size = grid**2
	if stage==2:
		attn = attn[:, 1::, 1::]

	B, H, W = attn.size()
	attn = attn.reshape(B, H, int(W**.5), int(W**.5))
	if attn.size(-1)>grid:
		attn = resize(attn)
	attn = attn.reshape(B,H,-1).permute(0,2,1)
	attn = attn.reshape(B,-1, int(H**.5), int(H**.5))
	if attn.size(-1)>grid:
		attn = resize(attn)

	attn = attn.reshape(B,new_size, new_size).permute(0,2,1)
	return attn

'''inspried by: https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py '''
def filter_attention_map(raw_attn, discard_ratio, head_fusion, show_fig=False):
	# raw_attn: B x h x N x N
	#import ipdb; ipdb.set_trace()

	H, W = raw_attn.size(-2), raw_attn.size(-1)

	ori_attn = torch.clone(raw_attn.detach())
	if head_fusion == 'mean':
		raw_attn = raw_attn.mean(axis=1)
	elif head_fusion == 'max':
		raw_attn = raw_attn.max(axis=1)[0]
	elif head_fusion == 'min':
		raw_attn = raw_attn.min(axis=1)[0]
	else:
		raise "head fusion type not supported"


	flat = raw_attn.view(raw_attn.size(0), -1)
	_, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
	indices_y, indices_x = indices / W, indices % W
	indices_y = indices_y.type(torch.LongTensor)
	indices_x = indices_x.type(torch.LongTensor)
	
	new_attn = flat.reshape(flat.size(0), H, W)
	new_attn[:, indices_y, indices_x] = 0

	if show_fig:
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots(1, 2)
		ax = ax.flat
		ax[0].imshow(ori_attn[0].mean(0).cpu().numpy() / ori_attn[0].mean(1).cpu().numpy().max())
		ax[1].imshow(new_attn[0].cpu().numpy() / new_attn[0].cpu().numpy().max())
		fig.savefig('rollout_filter_max.png')
		plt.close(fig)
		breakpoint()
	return new_attn


def get_attention_rollout(model, input, grid=7, use_res=True, display_map=False):
	# output attention scores of all layers
	#print('debug attn mats...')
	with torch.no_grad():
		#import ipdb; ipdb.set_trace()
		x, cls_tokens = model.both_forward(input)
		attn_mats = []

		resize = torch.nn.AdaptiveAvgPool2d((grid, grid))
		for si in range(0,3):
			stage = getattr(model, f'stage{si}')
			for i, blk in enumerate(stage.blocks):
				# raw_attn = blk._probs[0].mean(1) # original
				raw_attn = blk._probs[0]
				raw_attn = filter_attention_map(raw_attn, discard_ratio=0.1, head_fusion='min')
				#print(f'stage {si}, blk {i}, attn {raw_attn.size()}')
				resized_attn = resize_attn_map(raw_attn, resize, si, grid, blk_id=i)
				attn_mats.append(resized_attn.cpu().detach())

		attn_mats = torch.stack(attn_mats)

		if use_res:
			residual_att = torch.eye(attn_mats.size(2)).expand(attn_mats.size(0), attn_mats.size(1),-1,-1)
			attn_mats = attn_mats + residual_att
			attn_mats = attn_mats / attn_mats.sum(dim=-1).unsqueeze(-1)

	joint_attentions = []
	joint_attentions.append(attn_mats[0])
	for j in range(1, len(attn_mats)):
		joint_attentions.append(torch.bmm(attn_mats[j], joint_attentions[j-1]))

	if display_map:
		# visual_attention_rollout_images(input, joint_attentions, layer_id=-1, rand_pix=24)
		visual_attention_rollout_images_mean(input, joint_attentions, layer_id=-1)

	return joint_attentions

def get_vit_block_output(model, x, final_only=True):
	x = model.patch_embed(x)
	cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # (n_samples, 1, embed_dim)
	x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
	x = x + model.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
	x = model.pos_drop(x)
	if final_only:
		x = model.blocks(x)
		x = model.norm(x)
		return x
	else:
		xs = []
		for block in model.blocks:
			x = block(x)
			xs.append(x)
		xs = torch.stack(xs, 0)
		xs = xs.permute(1,0,2,3)[:,:,1::,:] # remove cls token
	
		return xs

def get_pretraind_res50(imagenet=True):
	model = models.resnet50(pretrained=False)
	if imagenet:
		model.load_state_dict(torch.load('/home/czhang/Pretrained_models/ResNet/resnet50-19c8e357.pth'), strict=False)
	return model

		
def evaluate_patch_similarity(model, dataset, dataloader):
	model.eval()
	final_iter = tqdm(dataloader, desc='Embedding Data...')
	sims = []
	for idx, inp in enumerate(final_iter):
		
		input_img, target = inp[1], inp[0]
		patches = get_vit_block_output(model, input_img.cuda(), final_only=False)
		b,L,n,d = patches.size()
		patches = F.normalize(patches, p=2, dim=-1)
		sim = torch.einsum('blmd,blnd->blmn', patches, patches)  # b L m n 
		diag_ids = torch.eye(n).expand(b,L,-1,-1).cuda()
		sim = sim - diag_ids
		avg_sim = torch.sum(sim, dim=(2,3)) / (n*(n-1))# b L
		sims.append(avg_sim.cpu().detach())
	sims = torch.cat(sims)
	nimg = sims.size(0)
	sims = torch.mean(sims, 0)
	print(f'evaluated on {nimg} images, and the similarity is {sims.cpu().detach().numpy()}')
	return sims

def evaluate(model, dataset, dataloader, training=False, trunc_nums=None, use_uniform=False, grid_size=4, use_inverse=False, temperature=1.0, use_cls_token=False, attn_blk_ind=0, use_ot=True, ot_part=0.1, to_submit=False, use_minus=False, use_rollout=False, plot_topk=1):

	device = torch.device('cuda')
	model.eval()
	no_training = not training
	use_featvit = not use_rollout

	use_cam = False
	show_self_sim = False
	save_dir = f'visual/{model.pars.dataset}_{model.pars.arch}_g{grid_size}'
	save_dir = save_dir + '_imagenet' if model.pars.not_pretrained else save_dir + '_fine'
	if use_uniform:
		save_dir += '_uniform'
	if use_inverse:
		save_dir += '_inverse' + f't_{temperature}'
	if use_cls_token:
		save_dir += '_classtoken'
	if use_featvit:
		save_dir += f'_attn{attn_blk_ind}'
	if use_ot:
		save_dir += '_ot'
	if use_rollout:
		save_dir += '_rollout'

	print(save_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)


	with torch.no_grad():

		if no_training:
			if (7 % grid_size == 0):
				resize = nn.AdaptiveAvgPool2d(grid_size)
			else:
				resize = nn.Sequential(
					nn.Upsample(grid_size * 4, mode='bilinear', align_corners=True),
					nn.AdaptiveAvgPool2d(grid_size),
				)

		target_labels = []
		feature_bank = []
		feature_bank_center = []
		cam_bank = []
		labels = []
		q_list = []
		rollout_list = []
		kt_list = []
		final_iter = tqdm(dataloader, desc='Embedding Data...')


		for idx, inp in enumerate(final_iter):
			# if idx>10: break
			input_img, target = inp[1].cuda(), inp[0]
			target_labels.extend(target.numpy().tolist())
			out = model(input_img)

			if not use_featvit:
				#q = get_qk(model.model, input_img, attn_blk_ind)
				rollout = get_attention_rollout(model.model, input_img, display_map=False)
				rollout_list.append(rollout[-1].mean(1).data.cpu().detach())
				#q = model.model.stage2.blocks[-1].q
				#q_list.append(q.data)
				#kt_list.append(kt.cpu().detach())

			if isinstance(out, tuple):
				out, aux_f = out


			if no_training:
				# need resize feature maps
				enc_out, no_avg_feat = aux_f

				no_avg_feat = model.model.head(no_avg_feat)
				no_avg_feat = no_avg_feat.permute(0,2,1) # bs x C x L
				no_avg_feat = no_avg_feat.view(no_avg_feat.size(0), -1, int(no_avg_feat.size(-1)**0.5), int(no_avg_feat.size(-1)**0.5))

				if no_avg_feat.size(-1) != grid_size:
					no_avg_feat = resize(no_avg_feat)

				no_avg_feat = no_avg_feat.view(no_avg_feat.size(0), no_avg_feat.size(1), -1) # bs x C x L

				feature_bank.append(no_avg_feat.data.cpu().detach())
				feature_bank_center.append(out.data.cpu().detach())
			else:
				# no need to resize
				global_enc = aux_f[0]
				no_avg_feat = out.view(out.size(0), out.size(1), -1)
				feature_bank.append(no_avg_feat.data)
				feature_bank_center.append(global_enc.data)

			labels.append(target)

		if not use_featvit:
			# q_list = np.concatenate(q_list)
			# q_list = torch.tensor(q_list)
			#q_list = torch.cat(q_list, dim=0)
			#kt_list = torch.cat(kt_list, dim=0)
			rollout_list = torch.cat(rollout_list, dim=0)

		if use_cam:
			cam_bank = torch.cat(cam_bank, dim=0)

		feature_bank = torch.cat(feature_bank, dim=0)
		feature_bank_center = torch.cat(feature_bank_center, dim=0)
		labels = torch.cat(labels, dim=0)
		N, C, R = feature_bank.size()

		feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1)
		feature_bank_center = torch.nn.functional.normalize(feature_bank_center, p=2, dim=1)
		

	feature_bank_center = feature_bank_center.to(device)
	trunc_nums = trunc_nums or [0, 5, 10, 50, 100, 500, 1000]

	overall_r1 = {k: 0.0 for k in trunc_nums}
	overall_rp = {k: 0.0 for k in trunc_nums}
	overall_mapr = {k: 0.0 for k in trunc_nums}


	for idx in trange(len(feature_bank)):

		anchor_center = feature_bank_center[idx]
		anchor = feature_bank[idx].to(anchor_center)

		if show_self_sim and (idx % 100 == 0):
			visual_patch_sim(dataset, idx, feature_bank[idx], save_dir=save_dir)
		

		approx_sim, uv = calc_similarity(None, anchor_center, None, feature_bank_center, 0)
		
		approx_sim[idx] = -100

		approx_tops = torch.argsort(approx_sim, descending=True)

		if max(trunc_nums) > 0:
			top_inds = approx_tops[:max(trunc_nums)]

			if use_featvit:
				sim, uv = calc_similarity(anchor, anchor_center, feature_bank[top_inds],
										  feature_bank_center[top_inds], stage=1,
										  use_uniform=use_uniform,
										  use_inverse=use_inverse,
										  temperature=temperature,
										  use_cls_token=use_cls_token,
										  ot_temp=0.05,
										  use_minus=use_minus,
										  ot_part=ot_part)
			elif use_rollout:
				# use rollout map
				sim, uv = calc_similarity_cvt_rollout(anchor_center, anchor, rollout_list[idx],
													  feature_bank_center[top_inds], feature_bank[top_inds],
													  rollout_list[top_inds], stage=1,
													  use_uniform=use_uniform,
													  use_ot=use_ot,
													  ot_part=ot_part)
			else:
				#use query proj
				sim, uv = calc_similarity_cvt(anchor_center, anchor, q_list[idx], feature_bank_center[top_inds], feature_bank[top_inds], q_list[top_inds], stage=1, use_uniform=use_uniform, use_ot=use_ot)


			rank_in_tops = torch.argsort(sim + approx_sim[top_inds], descending=True)

		for trunc_num in trunc_nums:
			if trunc_num == 0:
				final_tops = approx_tops
			else:
				rank_in_tops_real = top_inds[rank_in_tops][:trunc_num]

				final_tops = torch.cat([rank_in_tops_real, approx_tops[trunc_num:]], dim=0)

			r1, rp, mapr = get_metrics_rank(final_tops.data.cpu(), labels[idx], labels)


			overall_r1[trunc_num] += r1
			overall_rp[trunc_num] += rp
			overall_mapr[trunc_num] += mapr


		show_id = 0
		top_id = final_tops.data.cpu()[show_id]
		top_label = labels[top_id]
		query_label = labels[idx]
		top_rank_id = rank_in_tops[show_id]
		if plot_topk > 1:
			top_id = final_tops.data.cpu()[:plot_topk]
			top_label = labels[top_id]

		if ot_part<1:
			vis_dir = os.path.join(save_dir, f'part_{ot_part}')
		else:
			vis_dir = save_dir


		if idx<1000 and idx%10==0:
			visual_heatmap(dataset, idx, top_id, query_label, top_label, top_rank_id, uv,
						   save_dir=vis_dir,
						   temperature=temperature,
						   use_cls_token=use_cls_token,
						   to_submit=to_submit,

						   )

			#visual_self_cross_flow(dataset, idx, top_id, query_label, top_label, top_rank_id, uv, self_uv, save_dir=save_dir)
	

	for trunc_num in trunc_nums:
		overall_r1[trunc_num] /= float(N / 100)
		overall_rp[trunc_num] /= float(N / 100)
		overall_mapr[trunc_num] /= float(N / 100)

		print(f"trunc_num: {trunc_num}, ot part: {ot_part}")
		print('###########')
		print('Now rank-1 acc=%f, RP=%f, MAP@R=%f' % (overall_r1[trunc_num], overall_rp[trunc_num], overall_mapr[trunc_num]))

	data = {
		'r1': [overall_r1[k] for k in trunc_nums],
		'rp': [overall_rp[k] for k in trunc_nums],
		'mapr': [overall_mapr[k] for k in trunc_nums],
	}
	return data
