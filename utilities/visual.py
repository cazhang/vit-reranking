import torch
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.patches import ConnectionPatch
import cv2
from PIL import Image
import numpy as np
from os.path import join
from .diml import input_inv_transform
import torch.nn.functional as F

os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'

white=[255,255,255]
green=[0, 255, 0]
red=[255, 0, 0]

cmap = plt.get_cmap('jet')

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
	
def normalize_vector(x):
	assert x.ndim==1
	max_val, min_val = np.max(x), np.min(x)
	x = (x-min_val) / (max_val-min_val+np.finfo(np.float64).eps)
	return x

def visual_cross_correlation(cc, top_id, img, q_img, q_id):
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	from matplotlib import pyplot as plt
	N, R = cc.size()
	H, W = int(R**.5), int(R**.5)
	cc = cc.cpu().detach()
	fig, axs = plt.subplots(1, 4, figsize=(14, 6))
	axs = axs.flat
	att = cc[top_id].view(H,W)

	axs[0].imshow(q_img)
	axs[0].set_axis_off()
	axs[1].imshow(img)
	axs[1].set_axis_off()

	# im = axs[2].imshow(att)
	# divider = make_axes_locatable(axs[2])
	# cax = divider.append_axes('right', size='5%', pad=0.05)
	# cbar = fig.colorbar(im, cax=cax)
	# cbar.ax.tick_params(labelsize=10)

	im = axs[2].imshow(F.relu(att))
	axs[2].set_axis_off()
	divider = make_axes_locatable(axs[2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im, cax=cax)
	cbar.ax.tick_params(labelsize=10)

	im = axs[3].imshow(1-F.relu(att))
	axs[3].set_axis_off()
	divider = make_axes_locatable(axs[3])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im, cax=cax)
	cbar.ax.tick_params(labelsize=10)

	fig.savefig(f'cc_map{q_id}.png', bbox_inches='tight')
	plt.close(fig)

def draw_one_overlap(rgb, map, label, ax, patches=None, xcords=None, ycords=None, flows=None, sims=None):
	ax.imshow(np.uint8(np.clip(rgb * 0.5 + map * 0.5, 0, 255)))
	ax.text(0, 0, f'{label}')
	if patch is not None:
		for patch, x, y, t, s in zip(patches, xcords, ycords, flows, sims):
			ax.add_patch(patch)
			ax.text(x, y, '{0:.2f}x{1:.2f}'.format(t, s))
	ax.set_axis_off()

def visual_patch_sim(dataset, q_id, feature, save_dir):
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	C, R = feature.size()
	grid_size = int(R**0.5)
	cmap = plt.get_cmap('jet')
	sim = torch.einsum('cm,cs->ms', feature, feature).contiguous().view(R, R) # (49, 49)
	sim = sim.cpu().detach()
	fig, ax = plt.subplots(grid_size+1, grid_size, figsize=(20, 20))
	ax = ax.flat
	q_tensor = dataset.__getitem__(q_id)
	q_img = input_inv_transform(q_tensor[1].numpy())

	for rid in range(grid_size):
		for cid in range(grid_size):
			vid = rid*grid_size + cid
			patch_sim = sim[vid, :]
			patch_sim = patch_sim.view(grid_size, grid_size)
			ax[vid].imshow(patch_sim)
	ax[-1].imshow(q_img)

	save_name = os.path.join(save_dir, 'self_sim_{0:04d}.png'.format(q_id))
	fig.savefig(save_name)
	plt.close(fig)

def get_min_topk_ind(mat, topk=1):
	'''return the topk index of a matrix using argsort, ascending by default
	args:
		mat: matrix
	return:
		list of (y, x) coordinates'''
	hw_list = []
	ind = np.unravel_index(np.argsort(mat, axis=None), mat.shape)
	for k in range(topk):
		hw_list.append((ind[0][k], ind[1][k]))
	return hw_list

def get_patch_from_coord(ind_list, grid_size, patch_scale, sorting=False):
	'''get pyplot patch for a given coord from similarity matrix
	args:
	 list of (h, w), h is for target and w is for source
	 grid_size: height/width of image feature space
	 patch_scale: H // height
	return:
		patches_source (list), patches_tgt (list)
	'''
	width = grid_size * patch_scale
	colors = mcolors.BASE_COLORS
	if not isinstance(patch_scale, tuple):
		patch_scale_w = patch_scale_h = patch_scale
	else:
		patch_scale_w, patch_scale_h = patch_scale

	if sorting:
		by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
						 name)
						for name, color in colors.items())
		names = [name for hsv, name in by_hsv]
	else:
		names = [name for name, color in colors.items()]

	src_rect_list = []
	tgt_rect_list = []
	src_xy_list = []
	tgt_xy_list = []
	for ind, ind2d in enumerate(ind_list):
		src_coord = np.unravel_index(ind2d[1], (grid_size, grid_size))
		tgt_coord = np.unravel_index(ind2d[0], (grid_size, grid_size))

		src_x, src_y = src_coord[1] * patch_scale_w, src_coord[0] * patch_scale_h
		tgt_x, tgt_y = tgt_coord[1] * patch_scale_w, tgt_coord[0] * patch_scale_h

		color_ind = ind % len(names)

		src_rect = patches.Rectangle((src_x, src_y-1), width=patch_scale_w, height=patch_scale_h, linewidth=2, edgecolor=colors[names[color_ind]], fill=False)
		tgt_rect = patches.Rectangle((tgt_x, tgt_y-1), width=patch_scale_w, height=patch_scale_h, linewidth=2, edgecolor=colors[names[color_ind]], fill=False)
		src_rect_list.append(src_rect)
		tgt_rect_list.append(tgt_rect)
		src_xy_list.append((src_x, width-src_y))
		tgt_xy_list.append((tgt_x, width-tgt_y))
	return src_rect_list, tgt_rect_list, src_xy_list, tgt_xy_list



'''visual marginal distribution and representative patches'''
def visual_heatmap(dataset, q_id, top_id, q_label, top_label, top_rank_id, uv, save_dir, temperature, use_cls_token, to_submit=False):
	# u: spatial response of fb
	# v: spatial response of anchor
	# T: n_fb x n_anchor
	# sim: n_fb x n_anchor
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	cmap = plt.get_cmap('jet')
	border_size = 2
	if to_submit:
		new_dir = os.path.join(save_dir, f'query_{q_id:04d}')
		if not os.path.exists(new_dir):
			os.makedirs(new_dir)

	q_tensor = dataset.__getitem__(q_id)
	q_img = input_inv_transform(q_tensor[1].numpy())
	if len(top_id)==1:
		db_tensor = dataset.__getitem__(top_id)
		db_img = input_inv_transform(db_tensor[1].numpy())
	else:
		if to_submit:
			cv2.imwrite(os.path.join(new_dir, f'topk_{q_id:04d}.png'), q_img[:, :, ::-1])

		cmb_img = q_img
		cmb_img = cv2.copyMakeBorder(cmb_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=white)
		for j, db_id in enumerate(top_id):
			db_tensor = dataset.__getitem__(db_id)
			db_img = input_inv_transform(db_tensor[1].numpy())
			if top_label[j]==q_label:
				db_img = cv2.copyMakeBorder(db_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=green)
			else:
				db_img = cv2.copyMakeBorder(db_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=red)
			if to_submit:
				cv2.imwrite(os.path.join(new_dir, f'topk_{db_id:04d}.png'), db_img[:, :, ::-1])

			cmb_img = cv2.hconcat([cmb_img, db_img])
		# save preview
		cv2.imwrite(os.path.join(save_dir, f'topk_{q_id:04d}.png'), cmb_img[:,:,::-1])
		return;

	#if q_id == 2:
	#	breakpoint()
	u, v = uv[0].cpu().numpy(), uv[1].cpu().numpy()
	T = uv[2].cpu().numpy()
	sim_r = uv[3].cpu().numpy()
	cc = uv[4]

	#visual_cross_correlation(cc, top_rank_id, db_img, q_img, q_id)

	vmap, umap, tmap, sim_r = v[top_rank_id], u[top_rank_id], T[top_rank_id], sim_r[top_rank_id]
	grid_size = int(umap.shape[0]**0.5)
	patch_scale = 224 // grid_size
	#tmap = tmap * (grid_size**4)
	umap = normalize_vector(umap)
	vmap = normalize_vector(vmap)

	max_ids = get_min_topk_ind(-sim_r, topk=3)
	src_rect_list, tgt_rect_list, src_xy_list, tgt_xy_list = get_patch_from_coord(max_ids, grid_size, patch_scale)

	fig, ax = plt.subplots(2, 2, figsize=(10, 8))
	ax = ax.flat
	from copy import deepcopy
	src_rect_list2 = deepcopy(src_rect_list)
	tgt_rect_list2 = deepcopy(tgt_rect_list)
	vmap = cmap(vmap.reshape((grid_size, grid_size)))[:, :, :3] * 255
	vmap = cv2.resize(vmap, (224, 224), interpolation=cv2.INTER_LINEAR)
	q_out = np.uint8(np.clip(q_img*0.5+vmap*0.5,0,255))
	ax[0].imshow(q_out)
	ax[0].text(20, 0, f'{q_label}')
	# ax[0].add_patch(max_src_rect)
	# ax[0].add_patch(min_src_rect)
	# ax[0].text(max_src_x, max_src_y, '{0:.2f}x{1:.2f}'.format(max_t, max_sim))
	# ax[0].text(min_src_x, min_src_y, '{0:.2f}x{1:.2f}'.format(min_t, min_sim))
	ax[0].set_axis_off()
	for p in src_rect_list:
			ax[0].add_patch(p)

	umap = cmap(umap.reshape((grid_size,grid_size)))[:, :, :3] * 255
	umap = cv2.resize(umap, (224, 224), interpolation=cv2.INTER_LINEAR)
	db_out = np.uint8(np.clip(db_img*0.5+umap*0.5, 0, 255))
	ax[1].imshow(db_out)
	ax[1].text(20, 0, f'{top_label}')
	ax[1].set_axis_off()
	for p in tgt_rect_list:
			ax[1].add_patch(p)

	if False:
		for src_xy, tgt_xy in zip(src_xy_list, tgt_xy_list):
			con = ConnectionPatch(xyA=src_xy, xyB=tgt_xy, coordsA="data", coordsB="data", axesA=ax[1], axesB=ax[0], color="red")
			ax[1].add_artist(con)
	# ax[1].add_patch(max_tgt_rect)
	# ax[1].add_patch(min_tgt_rect)
	# ax[1].text(max_tgt_x, max_tgt_y, '{0:.2f}x{1:.2f}'.format(max_t, max_sim))
	# ax[1].text(min_tgt_x, min_tgt_y, '{0:.2f}x{1:.2f}'.format(min_t, min_sim))

	ax[2].imshow(tmap); ax[2].text(0,0, f'T')
	ax[3].imshow(sim_r); ax[3].text(0,0, f'Sim')

	if to_submit:
		fig1, ax1 = plt.subplots(1,1)
		ax1.imshow(q_out)
		ax1.set_axis_off()
		for p in src_rect_list2:
			ax1.add_patch(p)
		extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		fig1.savefig(join(new_dir, 'qmatch.png'), bbox_inches='tight')
		fig2, ax2 = plt.subplots(1, 1)
		ax2.imshow(db_out)
		ax2.set_axis_off()
		for p in tgt_rect_list2:
			ax2.add_patch(p)
		extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		fig2.savefig(join(new_dir, 'dmatch.png'), bbox_inches='tight')
		cv2.imwrite(join(new_dir, 'query.png'), q_img[:, :, ::-1])
		cv2.imwrite(join(new_dir, 'database.png'), db_img[:, :, ::-1])
		plt.close(fig1)
		plt.close(fig2)
	token_type = 'CLS' if use_cls_token else 'AVG'
	save_name = os.path.join(save_dir, f'heatmap_{q_id:04d}_{temperature:.2f}_{token_type}.png')
	fig.savefig(save_name)
	plt.close(fig)


'''visual marginal distribution and representative patches on MSLS'''
def visual_heatmap_msls(query_set, db_set, q_id, top_id, top_rank_id, uv, save_dir, 
use_cls_token, to_submit=False, city_num=0):
	# u: spatial response of fb
	# v: spatial response of anchor
	# T: n_fb x n_anchor
	# sim: n_fb x n_anchor
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	cmap = plt.get_cmap('jet')
	fig, ax = plt.subplots(2, 2, figsize=(10, 8))
	ax = ax.flat
	q_tensor = query_set.__getitem__(q_id)[0]
	q_img = input_inv_transform(q_tensor.numpy())

	H = q_img.shape[0]
	W = q_img.shape[1]
	u, v = uv[0].cpu().numpy(), uv[1].cpu().numpy()
	T = uv[2].cpu().numpy()
	sim_r = uv[3].cpu().numpy()
	cc = uv[4]
	
	vmap, umap, tmap, sim_r = v[top_rank_id], u[top_rank_id], T[top_rank_id], sim_r[top_rank_id]
	grid_size = int(umap.shape[0]**0.5)
	patch_scale = (W // grid_size, H // grid_size) # w, h
	sim_t = sim_r * tmap
	max_ids = get_min_topk_ind(-sim_t, topk=3)
	src_rect_list, tgt_rect_list, src_xy_list, tgt_xy_list = get_patch_from_coord(max_ids, grid_size, patch_scale)


	tmap = tmap * (grid_size**4)
	umap = normalize_vector(umap)
	vmap = normalize_vector(vmap)

	vmap = cmap(vmap.reshape((grid_size, grid_size)))[:, :, :3] * 255
	vmap = cv2.resize(vmap, (W, H), interpolation=cv2.INTER_LINEAR)
	ax[0].imshow(np.uint8(np.clip(q_img*0.5+vmap*0.5,0,255)))
	ax[0].set_axis_off()
	for p in src_rect_list:
		ax[0].add_patch(p)

	db_tensor = db_set.__getitem__(top_id)[0]
	db_img = input_inv_transform(db_tensor.numpy())
	umap = cmap(umap.reshape((grid_size,grid_size)))[:, :, :3] * 255
	umap = cv2.resize(umap, (W, H), interpolation=cv2.INTER_LINEAR)
	ax[1].imshow(np.uint8(np.clip(db_img*0.5+umap*0.5, 0, 255)))
	ax[1].set_axis_off()
	for p in tgt_rect_list:
		ax[1].add_patch(p)

	ax[2].imshow(q_img); ax[2].text(0,0, f'query')
	ax[3].imshow(db_img); ax[3].text(0,0, f'db')
	token_type = 'CLS' if use_cls_token else 'AVG'
	save_name = os.path.join(save_dir, f'heatmap_{token_type}_{city_num:01d}_query{q_id:04d}.png')
	fig.savefig(save_name)
	plt.close(fig)


'''visual marginal distribution for self and cross'''
def visual_self_cross_flow(dataset, q_id, top_id, q_label, top_label, top_rank_id, uv, self_uv, save_dir):
	# u: spatial response of fb
	# v: spatial response of anchor
	# T: n_fb x n_anchor
	# sim: n_fb x n_anchor
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	cmap = plt.get_cmap('jet')
	fig, ax = plt.subplots(2, 2, figsize=(10, 8))
	ax = ax.flat

	q_tensor = dataset.__getitem__(q_id)
	q_img = input_inv_transform(q_tensor[1].numpy())
	db_tensor = dataset.__getitem__(top_id)
	db_img = input_inv_transform(db_tensor[1].numpy())

	u, v = uv[0].cpu().numpy(), uv[1].cpu().numpy()
	self_u, self_v = self_uv[0].cpu().numpy(), self_uv[1].cpu().numpy()
	vmap, umap = v[top_rank_id], u[top_rank_id]
	self_vmap, self_umap = self_v[0], self_u[top_rank_id]

	grid_size = int(umap.shape[0]**0.5)
	patch_scale = 224//grid_size
	if True:
		umap = normalize_vector(umap)
		vmap = normalize_vector(vmap)
		self_umap = normalize_vector(self_umap)
		self_vmap = normalize_vector(self_vmap)

	vmap = cmap(vmap.reshape((grid_size, grid_size)))[:, :, :3] * 255
	vmap = cv2.resize(vmap, (224, 224), interpolation=cv2.INTER_LINEAR)
	ax[0].imshow(np.uint8(np.clip(q_img*0.5+vmap*0.5,0,255)))
	ax[0].text(0, 0, f'cross {q_label}')
	ax[0].set_axis_off()

	umap = cmap(umap.reshape((grid_size,grid_size)))[:, :, :3] * 255
	umap = cv2.resize(umap, (224, 224), interpolation=cv2.INTER_LINEAR)
	ax[1].imshow(np.uint8(np.clip(db_img*0.5+umap*0.5, 0, 255)))
	ax[1].text(0, 0, f'cross {top_label}')
	ax[1].set_axis_off()

	self_vmap = cmap(self_vmap.reshape((grid_size, grid_size)))[:, :, :3] * 255
	self_vmap = cv2.resize(self_vmap, (224, 224), interpolation=cv2.INTER_LINEAR)
	ax[2].imshow(np.uint8(np.clip(q_img * 0.5 + self_vmap * 0.5, 0, 255)))
	ax[2].text(0, 0, f'self {q_label}')
	ax[2].set_axis_off()

	self_umap = cmap(self_umap.reshape((grid_size, grid_size)))[:, :, :3] * 255
	self_umap = cv2.resize(self_umap, (224, 224), interpolation=cv2.INTER_LINEAR)
	ax[3].imshow(np.uint8(np.clip(db_img * 0.5 + self_umap * 0.5, 0, 255)))
	ax[3].text(0, 0, f'self {top_label}')
	ax[3].set_axis_off()

	save_name = os.path.join(save_dir, 'heatmap_{0:04d}.png'.format(q_id))
	fig.savefig(save_name)
	plt.close(fig)

def visual_attention_rollout_layers(input, joint_attentions, img_id=0, grid=7, rand_pix=-1):
	# visualise rollout multiple layers of one image
	fig, ax = plt.subplots(4, 4, figsize=(14, 8), constrained_layout=True)
	ax = ax.flat
	img = input_inv_transform(input[img_id].cpu().detach().numpy())
	H, W = img.shape[0], img.shape[1]
	patch_scale = H // grid
	for layer_id in range(len(joint_attentions)):
		rollouts = joint_attentions[layer_id]
		if rand_pix < 0:
			rand_pix = np.random.randint(0, grid**2-1)

		assert rand_pix < (grid**2)
		y, x = np.unravel_index(rand_pix, (grid, grid))
		x, y = x * patch_scale, y * patch_scale
		rect = patches.Rectangle((x, y - 1), patch_scale, patch_scale, linewidth=2, edgecolor='r', fill=False)

		rollout = rollouts[rand_pix, :]
		rollout = rollout.reshape(grid, grid).numpy()
		mask = cmap(rollout / rollout.max())[:, :, :3] * 255
		mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
		output = np.uint8(np.clip(img * 0.5 + mask * 0.5, 0, 255))
		#mask = cv2.resize(rollout / rollout.max(), (W, H))[..., np.newaxis]
		#output = (mask * img).astype("uint8")
		ax[layer_id].imshow(output)
		ax[layer_id].add_patch(rect)
	#ax[id+1].imshow(rollouts)
	save_name = f'rollout_img{img_id}_layers.png'
	fig.savefig(save_name)
	plt.close(fig)


def visual_attention_rollout_images(input, joint_attentions, layer_id=-1, grid=7, rand_pix=-1):
	# visualise rollout one layers of multiple image
	fig, ax = plt.subplots(4, 4, figsize=(14, 8), constrained_layout=True)
	ax = ax.flat
	rollouts = joint_attentions[layer_id]
	for img_id in range(input.shape[0]):
		img = input_inv_transform(input[img_id].cpu().detach().numpy())
		H, W = img.shape[0], img.shape[1]
		patch_scale = H // grid
		if rand_pix < 0:
			rand_pix = np.random.randint(0, grid ** 2 - 1)

		assert rand_pix < (grid ** 2)
		y, x = np.unravel_index(rand_pix, (grid, grid))
		x, y = x * patch_scale, y * patch_scale
		rect = patches.Rectangle((x, y - 1), patch_scale, patch_scale, linewidth=2, edgecolor='r', fill=False)

		rollout = rollouts[img_id, rand_pix, :]
		rollout = rollout.reshape(grid, grid).numpy()
		mask = cmap(rollout / rollout.max())[:, :, :3] * 255
		mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
		output = np.uint8(np.clip(img * 0.5 + mask * 0.5, 0, 255))
		# mask = cv2.resize(rollout / rollout.max(), (W, H))[..., np.newaxis]
		# output = (mask * img).astype("uint8")
		ax[img_id].imshow(output)
		ax[img_id].add_patch(rect)
	# ax[id+1].imshow(rollouts)
	save_name = f'rollout_max_layer{layer_id}_images.png'
	fig.savefig(save_name)
	plt.close(fig)

def visual_attention_rollout_images_mean(input, joint_attentions, layer_id=-1, grid=7):
	# visualise rollout one layers of multiple image
	fig, ax = plt.subplots(4, 4, figsize=(14, 8), constrained_layout=True)
	ax = ax.flat
	rollouts = joint_attentions[layer_id]
	for img_id in range(input.shape[0]):
		img = input_inv_transform(input[img_id].cpu().detach().numpy())
		H, W = img.shape[0], img.shape[1]
		rollout = rollouts[img_id].mean(0)
		rollout = rollout.reshape(grid, grid).numpy()
		if False:
			mask = rollout / rollout.max()
			heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
			heatmap = cv2.resize(heatmap, (W,H), interpolation=cv2.INTER_LINEAR)
			heatmap = np.float32(heatmap) / 255
			output = heatmap + np.float32(img) / 255
			output = output / np.max(output)
			output = np.uint8(output * 255)
		else:
			mask = cmap(rollout / rollout.max())[:, :, :3] * 255
			mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
			output = np.uint8(np.clip(img * 0.5 + mask * 0.5, 0, 255))

		ax[img_id].imshow(output)

	# ax[id+1].imshow(rollouts)
	save_name = f'rollout_max_layer{layer_id}_images_mean.png'
	fig.savefig(save_name)
	plt.close(fig)

