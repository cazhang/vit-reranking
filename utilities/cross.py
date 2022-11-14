import torch
from torch import nn, einsum
import torch.nn.functional as F

import os
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image
import numpy as np
os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

def calc_similarity_cross(anchor, feature_bank, model):
	# anchor: 1 x L x D
	# feature_banck: N x L x D
	# model: cross-vit using class token as probe
	assert feature_bank.ndim==3 and anchor.ndim==3
	N, _, D = feature_bank.size()
	anchors = anchor.expand(N, -1, -1).cuda()
	feature_bank = feature_bank.cuda()
	anchor_class, fb_class = model(anchors, feature_bank)
	sim = torch.einsum('bd,bd->b', anchor_class, fb_class)
	return sim

def get_vit_features(model, x, opt):
	x = model.patch_embed(x)
	cls_token = model.cls_token.expand(x.shape[0], -1, -1)  # (n_samples, 1, embed_dim)
	x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
	x = x + model.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
	x = model.pos_drop(x)
	x = model.blocks(x)
	feats = x
	x = model.head(model.norm(x[:,0]))
	if 'normalize' in opt.arch:
		x = torch.nn.functional.normalize(x, dim=-1)
	return x, feats

def get_embed_and_token(model, input, opt):
	if opt.skip_last_vit_norm:
		embeds = get_vit_features(model.basenet.model, input, opt)
		if isinstance(embeds, tuple): embeds, features = embeds
	else:
		embeds = model.basenet(input)
		if isinstance(embeds, tuple): embeds, (avg_features, features) = embeds

		cls_token = avg_features.unsqueeze(1)
		patch_token = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
		features = torch.cat((cls_token, patch_token), dim=1)
	return embeds, features

### for cross-cvt
def normalize_all(x, y, x_mean, y_mean):
	x = F.normalize(x, dim=1)
	y = F.normalize(y, dim=1)
	x_mean = F.normalize(x_mean, dim=1)
	y_mean = F.normalize(y_mean, dim=1)
	return x, y, x_mean, y_mean

def Sinkhorn(K, u, v, max_iter):
	r = torch.ones_like(u)
	c = torch.ones_like(v)
	thresh = 1e-1
	for i in range(max_iter):
		r0 = r
		r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
		c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
		err = (r - r0).abs().mean()
		if err.item() < thresh:
			break

	T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

	return T

def cross_attention(x, y, x_mean, y_mean):
	N, C = x.shape[:2]
	x = x.view(N, C, -1)
	y = y.view(N, C, -1)

	att = F.relu(torch.einsum("nc,ncr->nr", x_mean, y)).view(N, -1)
	u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
	att = F.relu(torch.einsum("nc,ncr->nr", y_mean, x)).view(N, -1)
	v = att / (att.sum(dim=1, keepdims=True) + 1e-5)
	return u, v


def pair_wise_wdist(x, y, eps=0.05, max_iter=100, use_uniform=False):
	B, C, H, W = x.size()
	x = x.view(B, C, -1, 1)
	y = y.view(B, C, 1, -1)
	x_mean = x.mean([2, 3])
	y_mean = y.mean([2, 3])

	x, y, x_mean, y_mean = normalize_all(x, y, x_mean, y_mean)
	dist1 = torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=1) + 1e-6).view(B, H*W, H*W)
	dist2 = torch.sqrt(torch.sum(torch.pow(x_mean - y_mean, 2), dim=1) + 1e-6).view(B)

	x = x.view(B, C, -1)
	y = y.view(B, C, -1)

	# order???
	sim = torch.einsum('bcs, bcm->bms', x, y).contiguous()

	if use_uniform:
		u = torch.zeros(B, H*W, dtype=sim.dtype, device=sim.device).fill_(1. / (H * W))
		v = torch.zeros(B, H*W, dtype=sim.dtype, device=sim.device).fill_(1. / (H * W))
	else:
		u, v = cross_attention(x, y, x_mean, y_mean)

	wdist = 1.0 - sim.view(B, H*W, H*W)

	with torch.no_grad():
		K = torch.exp(-wdist / eps)
		T = Sinkhorn(K, u, v, max_iter)

	if torch.isnan(T).any():
		return None

	dist1 = torch.sum(T * dist1, dim=(1, 2))
	dist = dist1 + dist2
	dist = dist / 2

	return dist