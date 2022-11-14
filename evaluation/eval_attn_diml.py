import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm, trange
from utilities.diml import Sinkhorn, calc_similarity, calc_similarity_featvit
from utilities.visual import visual_heatmap, visual_self_cross_flow, input_inv_transform
from utilities.cam import get_cam_ouput, demo
from pytorch_grad_cam.utils.image import show_cam_on_image
from evaluation.metrics import get_metrics_rank, get_metrics

os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'

# Unlike DIML which uses final layer feature for cross correlation, this code use attention map from vit for optimal transport computation

def get_vit_qk(model, x, blk_ind=0):
	## get q and k of pretrained vit. Fix: should go through previous blocks
	x = model.patch_embed(x)
	cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # (n_samples, 1, embed_dim)
	x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
	x = x + model.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
	x = model.pos_drop(x)

	for i, blk in enumerate(model.blocks):
		if i < blk_ind:
			x = blk(x)
		else:
			# last blk
			x = blk.norm1(x)
			B,N,C = x.shape
			qkv = blk.attn.qkv(x).reshape(B, N, 3, blk.attn.num_heads, C//blk.attn.num_heads)
			q = qkv[:, :, 0] # bs x patch x head x dim
			q = q.permute(0,2,1,3).contiguous() # bs x head x patch x dim 
			k = qkv[:, :, 1] # bs x patch x head x dim
			k = k.permute(0,2,1,3).contiguous() # bs x head x dim x patch
	return q, k

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
	
def save_cam_map(imgs, cams, inds, save_dir=None):
    if not save_dir:
        save_dir = './cam_visual'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for ind, name_id in enumerate(inds):
        cam, img = cams[ind], imgs[ind]
  
        img = input_inv_transform(img.cpu().detach().numpy()) / 255.0
        cam_image = show_cam_on_image(img, cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        save_name = os.path.join(save_dir, f'{name_id}_cam.png')
        cv2.imwrite(save_name, cam_image)
		
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

def evaluate(model, dataset, dataloader, training=False, trunc_nums=None, use_uniform=False, grid_size=4,
			 use_inverse=False, temperature=1.0, use_cls_token=False, attn_blk_ind=0, use_ot=True,
			 ot_part=0.1, to_submit=False, use_minus=False, use_rollout=False, use_soft=False):
	model.eval()
	no_training = not training
	use_pre=False
	use_featvit = True
	use_cam = False
	if use_cam:
		res50 = get_pretraind_res50(imagenet=True)

	save_dir = f'/home/czhang/DIML/visual/{model.pars.dataset}_{model.pars.arch}_g{grid_size}'
	save_dir = save_dir + '_imagenet' if model.pars.not_pretrained else save_dir + '_fine'
	if use_uniform:
		save_dir += '_uniform'
	if use_inverse:
		save_dir += '_inverse' + f't_{temperature}'
	if use_cls_token:
		save_dir += '_classtoken'
	if not use_featvit:
		save_dir += f'_attn{attn_blk_ind}'
	if use_ot:
		save_dir += '_ot'
	if use_rollout:
		save_dir += '_rollout'

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# with torch.no_grad():
	if no_training:
		if grid_size == 7:
			resize = nn.AdaptiveAvgPool2d(grid_size)
		else:
			resize = nn.Sequential(
				nn.Upsample(grid_size * 4, mode='bilinear', align_corners=True),
				nn.AdaptiveAvgPool2d(grid_size),
			)
		feature_bank_center = []

	target_labels = []
	feature_bank = []
	cam_bank = []
	labels = []
	q_list = []
	kt_list = []
	final_iter = tqdm(dataloader, desc='Embedding Data...')
	for idx, inp in enumerate(final_iter):
		#if idx>10: break
		input_img, target = inp[1].cuda(), inp[0]
		target_labels.extend(target.numpy().tolist())
		out = model(input_img)
		
		if use_cam:
			cam_map = get_cam_ouput(res50, input_img, cam_method='gradcam', use_cuda=True)
			cam_bank.append(torch.tensor(cam_map))
		#save_cam_map(input_img, cam_map, inp[2])
		
		if not use_featvit:
			q, kt = get_vit_qk(model.model, input_img)
			q_list.append(q.cpu().detach())
			kt_list.append(kt.cpu().detach())

		if isinstance(out, tuple):
			out, aux_f = out

		if no_training:
			enc_out, no_avg_feat = aux_f

			if not use_pre:
				# no_avg_feat = no_avg_feat.transpose(1, 3)
				no_avg_feat = model.model.head(no_avg_feat)
				no_avg_feat = no_avg_feat.permute(0, 2, 1)
				no_avg_feat = no_avg_feat.view(no_avg_feat.size(0), no_avg_feat.size(1), int(no_avg_feat.size(-1)**0.5), int(no_avg_feat.size(-1)**0.5))

				if no_avg_feat.size(-1) > grid_size:
					no_avg_feat = resize(no_avg_feat)

				no_avg_feat = no_avg_feat.view(no_avg_feat.size(0), no_avg_feat.size(1), -1)  # bs x C x L
			else:
				out = enc_out

			#out = out.unsqueeze(-1) # bs x 128 x 1
			no_avg_feat = no_avg_feat.contiguous().view(no_avg_feat.size(0), no_avg_feat.size(1), -1) # bs x 128 x 196
			#no_avg_feat = torch.cat((out, no_avg_feat), dim=-1) # bsx 128x 197
			feature_bank.append(no_avg_feat.data)
			feature_bank_center.append(out.data)
		else:
			feature_bank.append(out.cpu().detach())

		labels.append(target)

	if not use_featvit:
		q_list = torch.cat(q_list, dim=0)
		kt_list = torch.cat(kt_list, dim=0)
	if use_cam:
		cam_bank = torch.cat(cam_bank, dim=0)
		
	feature_bank = torch.cat(feature_bank, dim=0)
	feature_bank_center = torch.cat(feature_bank_center, dim=0)
	
	labels = torch.cat(labels, dim=0)
	N, C, R = feature_bank.size()

	feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1)
	feature_bank_center = torch.nn.functional.normalize(feature_bank_center, p=2, dim=1)

	trunc_nums = trunc_nums or [0, 5, 10, 50, 100, 500, 1000]

	overall_r1 = {k: 0.0 for k in trunc_nums}
	overall_rp = {k: 0.0 for k in trunc_nums}
	overall_mapr = {k: 0.0 for k in trunc_nums}

	print(f'saving to {save_dir}')

	for idx in trange(len(feature_bank)):

		anchor = feature_bank[idx]
		anchor_center = feature_bank_center[idx]
		
		if use_cam:
			anchor_cam = cam_bank[idx]

		if use_featvit:
			approx_sim, uv = calc_similarity(None, anchor_center, None, feature_bank_center, 0)
		else:
			approx_sim, uv = calc_similarity_vit(anchor_center, None, None, feature_bank_center, None, None, 0)


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
										  ot_part=ot_part,
										  use_soft=use_soft)
				# _, self_uv = calc_similarity_featvit(anchor, feature_bank[top_inds], 1, use_uniform, use_self=False)
			else:
				sim, uv = calc_similarity_vit(anchor_center, anchor, q_list[idx],
											  feature_bank_center[top_inds], feature_bank[top_inds],
											  q_list[top_inds], stage=1,
											  use_uniform=use_uniform, use_exp=use_exp)

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

		import random
		show_id = 0
		top_id = final_tops.data.cpu()[show_id]
		top_label = labels[top_id]
		query_label = labels[idx]
		top_rank_id = rank_in_tops[show_id]
		if ot_part<1:
			vis_dir = os.path.join(save_dir, f'part_{ot_part}')
		else:
			vis_dir = save_dir

		if idx < 100:
			visual_heatmap(dataset, idx, top_id, query_label, top_label, top_rank_id, uv,
						   save_dir=save_dir,
						   temperature=temperature,
						   use_cls_token=use_cls_token,
						   to_submit=to_submit
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
