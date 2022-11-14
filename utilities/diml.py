import torch
import torch.nn.functional as F
import os
import numpy as np

os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'

def visual_cross_correlation(cc):
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	from matplotlib import pyplot as plt
	N, R = cc.size()
	H, W = int(R**.5), int(R**.5)
	cc = cc.cpu().detach()
	fig, axs = plt.subplots(2, 2, figsize=(10, 8))
	axs = axs.flat
	att = cc[0].view(H,W)

	im = axs[0].imshow(att)
	divider = make_axes_locatable(axs[0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(im, cax=cax)

	im1 = axs[1].imshow(F.relu(att))
	divider = make_axes_locatable(axs[1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(im1, cax=cax)

	im2 = axs[2].imshow(1-F.relu(att))
	divider = make_axes_locatable(axs[2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(im2, cax=cax)

	im3 = axs[3].imshow(-F.relu(-att))
	divider = make_axes_locatable(axs[3])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	fig.colorbar(im3, cax=cax)


	fig.savefig('cc_map.png')
	plt.close(fig)

def Sinkhorn(K, u, v, iter=100):
	r = torch.ones_like(u)
	c = torch.ones_like(v)
	thresh = 1e-1
	for _ in range(iter):
		r0 = r
		r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
		c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
		err = (r - r0).abs().mean()
		if err.item() < thresh:
			break
	T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
	return T

'''sinkhorn algo supporting 1 dummy point
	inspired by: https://arxiv.org/pdf/2002.08276.pdf and adapted from
	https://pythonot.github.io/_modules/ot/partial.html#partial_wasserstein'''
def Sinkhorn_partial(K, u, v, ot_part=0.1):
	assert ot_part < 1 and ot_part >= 0
	b, m, n = K.shape
	bin = K.new_tensor(1-ot_part)
	bin0 = bin.expand(b,1)# b
	bins0 = bin.expand(b, 1, n)
	bin1 = bin.expand(b,1) # b
	bins1 = bin.expand(b, m, 1)
	# alpha = torch.max(K) * 1e5
	alpha = K.new_tensor(0.)
	alpha = alpha.expand(b, 1, 1)

	u_extended = torch.cat([u, bin0], -1)
	v_extended = torch.cat([v, bin1], -1)
	K_extended = torch.cat([torch.cat([K, bins1], -1), torch.cat([bins0, alpha], -1)], 1)
	T_extended = Sinkhorn(K_extended, u_extended, v_extended)
	return T_extended

def calc_similarity(anchor, anchor_center, fb, fb_center, stage, use_uniform=False, use_inverse=False, temperature=1.0, use_cls_token=False, ot_temp=0.05, use_minus=False, ot_part=1.0, use_soft=False):

	use_full = ot_part > 0.999
	if use_minus:
		use_inverse = False

	if stage == 0:
		sim = torch.einsum('c,nc->n', anchor_center, fb_center)
		return sim, None
	else:
		if use_cls_token:
			assert anchor_center.ndim==1
		else:
			anchor_center = torch.mean(anchor, dim=1)
			fb_center = torch.mean(fb, dim=-1)

		fb_center = fb_center.to(anchor_center)
		fb = fb.to(anchor_center)
		anchor_center=torch.nn.functional.normalize(anchor_center, p=2, dim=-1)
		fb_center=torch.nn.functional.normalize(fb_center, p=2, dim=-1)

		N, _, R = fb.size()

		sim = torch.einsum('cm,ncs->nsm', anchor, fb).contiguous().view(N, R, R)
		dis = 1.0 - sim
		K = torch.exp(-dis / ot_temp)
		cc = None
		if use_uniform:
			u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
			v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
		elif use_inverse:
			att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
			att = torch.exp(-att / temperature)
			u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
			att = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
			att = torch.exp(-att / temperature)
			v = att / (att.sum(dim=1, keepdims=True) + 1e-5)
		elif use_minus:
			cc = torch.einsum("c,ncr->nr", anchor_center, fb).view(N,R)
			att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
			att = 1-att
			u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
			att = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
			att = 1-att
			v = att / (att.sum(dim=1, keepdims=True) + 1e-5)
		elif use_soft:
			att = F.softmax(torch.einsum("c,ncr->nr", anchor_center, fb), -1).view(N, R)
			u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
			cc = torch.einsum("cr,nc->nr", anchor, fb_center).view(N, R)
			att = F.softmax(torch.einsum("cr,nc->nr", anchor, fb_center), -1).view(N, R)
			v = att / (att.sum(dim=1, keepdims=True) + 1e-5)
		else:
			att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
			u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
			cc = torch.einsum("cr,nc->nr", anchor, fb_center).view(N, R)
			att = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
			v = att / (att.sum(dim=1, keepdims=True) + 1e-5)

		if use_full:
			T = Sinkhorn(K, u, v)
		else:
			T_ext = Sinkhorn_partial(K, u, v, ot_part=ot_part)
			T = T_ext[:, :R, :R]

		#print(u.sum(), v.sum(), T.sum())
		sim_r = T*sim
		sim = torch.sum(sim_r, dim=(1, 2))
		if use_full:
			return sim, (u, v, T, sim_r, cc)
		else:
			return sim, (u,v,T_ext, sim_r, cc)

def calc_distance(anchor, anchor_center, fb, fb_center, stage, use_uniform=False, use_exp=True, temperature=1.0, use_cls_token=False):

	if stage == 0:
		dist = torch.sqrt(torch.sum(torch.pow(anchor_center - fb_center, 2), dim=1) + 1e-6).view(fb_center.size(0))
		#dist = torch.einsum('c,nc->n', anchor_center, fb_center)
		return dist, None
	else:

		N, C, R = fb.size()
		assert anchor.size(0)==C and anchor.size(1)==R
		if use_cls_token:
			assert anchor_center.ndim==1
		else:
			anchor_center = torch.mean(anchor, dim=-1)
			fb_center = torch.mean(fb, dim=-1)

		anchor_center=torch.nn.functional.normalize(anchor_center, p=2, dim=-1)
		fb_center=torch.nn.functional.normalize(fb_center, p=2, dim=-1)

		anchor = torch.nn.functional.normalize(anchor, p=2, dim=0)
		fb = torch.nn.functional.normalize(fb, p=2, dim=1)

		sim = torch.einsum('cm,ncs->nms', anchor, fb).contiguous().view(N, R, R)

		anchor = anchor.view(1, C, -1, 1) # 1 C 1 R
		fb = fb.view(N, C, 1, -1)
		dist = torch.sqrt(torch.sum(torch.pow(anchor - fb, 2), dim=1) + 1e-6).view(N, R, R)

		wdist = 1.0 - sim
		K = torch.exp(-wdist / 0.05)

		fb = fb.view(N, C, R)
		anchor = anchor.view(C, R)

		if use_uniform:
			u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
			v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
		elif use_exp:
			att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
			att = torch.exp(-att / temperature)
			u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
			att = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
			att = torch.exp(-att / temperature)
			v = att / (att.sum(dim=1, keepdims=True) + 1e-5)
		else:
			att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
			u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
			att = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
			v = att / (att.sum(dim=1, keepdims=True) + 1e-5)

		T = Sinkhorn(K, v, u)
		dist = torch.sum(T * dist, dim=(1, 2))
		T = T.permute(0,2,1)
		sim = sim.permute(0,2,1)
		return dist, (u,v,T,sim)


def calc_similarity_vit(anchor_center, anchor_feat, anchor_query, fb_center, fb_feat, fb_keyt, stage,
						use_uniform=False, use_exp=False, temperature=1.):
	# vit attention map
	if stage == 0:
		sim = torch.einsum('c,nc->n', anchor_center, fb_center)
		return sim, None
	else:
		anchor = anchor_feat
		fb = fb_feat
		N, _, R = fb.size()

		sim = torch.einsum('cm,ncs->nsm', anchor, fb).contiguous().view(N, R, R)
		sim_list = []
		## achnor_query: list()

		if not isinstance(anchor_query, list):
			anchor_query = [anchor_query]
			fb_keyt = [fb_keyt]
			
		for q, k in zip(anchor_query, fb_keyt):
			q=q.to(sim.device)
			k=k.to(sim.device)
			# q: nhead x ntoken x ndim

			q = torch.mean(q, dim=0) # ntoken x ndim
			k = torch.mean(k, dim=1) # bs x ntoken x ndim
			q = torch.nn.functional.normalize(q, p=2, dim=1)
			k = torch.nn.functional.normalize(k, p=2, dim=2)
			
			dp = torch.einsum('mc,nsc->nsm', q, k).contiguous().view(N, R+1, R+1) * (1/8)
			 
			#dp = q@kt #  bs x ntoken x ntoken
			#dp = dp.permute(0,2,1)
			dist = 1-dp[:,1::, 1::]
			K = torch.exp(-dist / 0.05)
			if use_uniform:
				u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
				v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
			elif use_exp:
				att = F.relu(dp[:,1::, 0]).view(N, R) # fb weights
				att = torch.exp(-att / temperature)
				u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
				att = F.relu(dp[:,0, 1::]).view(N, R) # anchor weights
				att = torch.exp(-att / temperature)
				v = att / (att.sum(dim=1, keepdims=True) + 1e-5)
			else:
				u = F.relu(dp[:,1::, 0]).view(N, R) # fb weights
				u = u / (u.sum(dim=1, keepdims=True) + 1e-5)
				v = F.relu(dp[:,0, 1::]).view(N, R) # anchor weights
				v = v / (v.sum(dim=1, keepdims=True) + 1e-5)
			T = Sinkhorn(K, u, v)
			sim_r = T * sim
			sim_list.append(torch.sum(T * sim, dim=(1,2)))
		
		sim = torch.cat(sim_list, dim=0)
		if sim.ndim>1:
			sim = torch.mean(sim, dim=0)
		return sim, (u,v,T,sim_r)


def calc_similarity_cvt(anchor_center, anchor, anchor_query, 
						fb_center, fb, fb_key, stage, use_uniform=False,
						use_ot=False):
	# vit attention map: max (D * T), where D_ij = 1-S_ij, and T_ij is OT computed from cross attn
	if stage == 0:
		sim = torch.einsum('c,nc->n', anchor_center, fb_center)
		return sim, None
	else:

		N, _, R = fb.size()
		
		sim_list = []
		sim = torch.einsum('cm,ncs->nsm', anchor, fb).contiguous().view(N, R, R) # fb x anchor

		if not isinstance(anchor_query, list):
			anchor_query = [anchor_query]
			fb_key = [fb_key]
			
		for q, k in zip(anchor_query, fb_key):

			q = torch.mean(q, dim=0) # ntoken x ndim
			k = torch.mean(k, dim=1) # bs x ntoken x ndim
			q = torch.nn.functional.normalize(q, p=2, dim=-1)
			k = torch.nn.functional.normalize(k, p=2, dim=-1)
			#kt = k.permute(0,2,1)
			dp = torch.einsum('mc,nsc->nsm', q, k).contiguous().view(N, R+1, R+1)
			dp_patch = dp[:,1::,1::]
			#dp = q@kt #  bs x ntoken x ntoken
			#dp = dp.permute(0,2,1)

			if use_ot:
				dist = 1-dp_patch
				K = torch.exp(-dist / 0.05)
				if use_uniform:
					u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
					v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
				else:
					u = F.relu(dp[:,1::,0]).view(N, R)
					u = u / (u.sum(dim=1, keepdims=True) + 1e-5)
					v = F.relu(dp[:,0,1::]).view(N, R)
					v = v / (v.sum(dim=1, keepdims=True) + 1e-5)

				T = Sinkhorn(K, u, v)
			else:
				u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
				v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
				T = F.softmax(dp_patch, dim=-1) * F.softmax(dp_patch, dim=-2)

			sim_r = T * sim
			sim_list.append(torch.sum(T * sim, dim=(1,2)))
		
		sim = torch.cat(sim_list, dim=0)
		if sim.ndim>1:
			sim = torch.mean(sim, dim=0)
		return sim, (u,v,T,sim_r)


def calc_similarity_cvt_rollout(anchor_center, anchor, anchor_query,
						fb_center, fb, fb_key, stage, use_uniform=False,
						ot_temp=0.05, use_ot=True, ot_part=1.0, device = torch.device('cuda')):
	# cvt attention rollout as marginal u, v, similar to CAM
	use_full = ot_part > 0.999
	if stage == 0:
		sim = torch.einsum('c,nc->n', anchor_center, fb_center)
		return sim, None
	else:

		anchor_query = anchor_query.to(anchor_center)
		fb_center = fb_center.to(anchor_center)
		fb = fb.to(anchor_center)
		fb_key = fb_key.to(anchor_center)
		N, _, R = fb.size()
		sim_list = []
		sim = torch.einsum('cm,ncs->nsm', anchor, fb).contiguous().view(N, R, R)  # fb x anchor

		dis = 1.0 - sim
		K = torch.exp(-dis / ot_temp)
		cc = None
		if use_uniform:
			u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
			v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)

		else:
			fb_key = fb_key.to(sim.device)
			anchor_query = anchor_query.to(sim.device)
			att = F.relu(fb_key).view(N, R)
			u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
			att = F.relu(anchor_query.expand(N,-1)).view(N, R)
			v = att / (att.sum(dim=1, keepdims=True) + 1e-5)

		if use_full:
			T = Sinkhorn(K, u, v)
		else:
			T_ext = Sinkhorn_partial(K, u, v, ot_part=ot_part)
			T = T_ext[:, :R, :R]
		sim_r = T*sim
		sim = torch.sum(sim_r, dim=(1, 2))
		if use_full:
			return sim, (u, v, T, sim_r,cc)
		else:
			return sim, (u,v,T_ext, sim_r,cc)

def calc_similarity_featvit(anchor_feat, fb_feat, stage, use_uniform=False, use_self=False, use_cam=False,
							anchor_cam=None, fb_cam=None):
	# vit feature map
	if stage == 0:
		anchor_center = anchor_feat[:, 0]
		fb_center = fb_feat[:, :, 0]
		sim = torch.einsum('c,nc->n', anchor_center, fb_center)
		return sim, None
	else:
		if use_cam:
			assert (anchor_cam is not None and fb_cam is not None), 'CAM map is none'
		
			size_cam = fb_cam.size()
		anchor_center = anchor_feat[:, 0]
		fb_center = fb_feat[:, :, 0]
		#anchor_center = anchor_feat[:,1::].mean(1)
		#fb_center = fb_feat[:, :, 1::].mean(2)
		anchor = anchor_feat[:, 1::]
		fb = fb_feat[:, :, 1::]
		N, _, R = fb.size()
		sqrt_R = int(R**0.5)
		resize = torch.nn.AdaptiveAvgPool2d(sqrt_R)

		sim = torch.einsum('cm,ncs->nsm', anchor, fb).contiguous().view(N, R, R)
		dis = 1-sim
		K = torch.exp(-dis / 0.05)

		if use_uniform:
			u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
			v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
		else:
			if use_self:
				u = F.relu(torch.einsum("nc,ncr->nr", fb_center, fb)).view(N, R)
				u = u / (u.sum(dim=1, keepdims=True) + 1e-5)
				v = F.relu(torch.einsum("c,cr->r", anchor_center, anchor)).view(1, R)
				v = v / (v.sum(dim=1, keepdims=True) + 1e-5)
				
			elif use_cam:
				#breakpoint()
				fb_cam = fb_cam.view(N,1,size_cam[1],size_cam[2])
				u = resize(fb_cam).view(N,R).to(sim.device)
				u = u / (u.sum(dim=1, keepdims=True) + 1e-5)
				anchor_cam = anchor_cam.expand(N,1,-1,-1)
				v = resize(anchor_cam).view(N,R).to(sim.device)
				v = v / (v.sum(dim=1, keepdims=True) + 1e-5)
			
			else:
				u = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
				u = u / (u.sum(dim=1, keepdims=True) + 1e-5)
				v = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
				v = v / (v.sum(dim=1, keepdims=True) + 1e-5)
		T = Sinkhorn(K, u, v)
		sim_r = torch.clone(sim.detach())
		sim = torch.sum(T * sim, dim=(1, 2))
		return sim, (u, v, T, sim_r)

def calc_similarity_mhvit(anchor_feat, fb_feat, stage, use_uniform=False):
	# vit multi-head feature map
	if stage == 0:
		anchor_center = anchor_feat[:, 0]
		fb_center = fb_feat[:, :, 0]
		sim = torch.einsum('c,nc->n', anchor_center, fb_center)
		return sim, None
		
	else:
		nhead = 12
		ndim = 64
		anchor_center = anchor_feat[:, 0]
		fb_center = fb_feat[:, :, 0]
		anchor = anchor_feat[:, 1::]
		fb = fb_feat[:, :, 1::]
		N, _, R = fb.size()

		anchor_center = anchor_center.reshape(nhead, ndim)
		fb_center = fb_center.reshape(-1, nhead, ndim)
		anchor = anchor.reshape(nhead, ndim, -1)
		fb = fb.reshape(N, nhead, ndim, -1)

		sim = torch.einsum('hdm,nhds->nhsm', anchor, fb).contiguous().view(N, nhead, R, R)
		dis = 1-sim
		K = torch.exp(-dis / 0.05)

		if use_uniform:
			u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
			v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
		else:
			att_col = F.relu(torch.einsum("hd,nhdr->nhr", anchor_center, fb)).view(N, nhead,R)
			u = att_col / (att_col.sum(dim=2, keepdims=True) + 1e-5)
			att_row = F.relu(torch.einsum("hdr,nhd->nhr", anchor, fb_center)).view(N, nhead,R)
			v = att_row / (att_row.sum(dim=2, keepdims=True) + 1e-5)

		sim_list = []
		for i in range(nhead):
			sim_one = sim[:,i,:,:]
			K_one = K[:,i,:,:]
			u_one = u[:,i,:]
			v_one = v[:,i,:]
			T_one = Sinkhorn(K_one, u_one, v_one)
			sim_list.append(T_one * sim_one)
		
		sim_final = torch.stack(sim_list)
		sim_final = torch.max(sim_final, dim=0).values
		sim_final = torch.sum(sim_final, dim=(1,2))

		return sim_final, (u, v)


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
