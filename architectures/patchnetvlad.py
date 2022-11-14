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

We thank Nanne https://github.com/Nanne/pytorch-NetVlad for the original design of the NetVLAD
class which in itself was based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
In our version we have significantly modified the code to suit our Patch-NetVLAD approach.

This is the key file that actually produces the Patch-NetVLAD features.

Currently we only support square patches, but this can be changed if needed by an end-user by
adjusting patchSize and patchStride to be a tuple of two ints (height, width). Any number of patch
sizes can be used, however very large numbers of patch sizes may exceed the available GPU memory.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
import faiss
import os
import numpy as np

from architectures.netvlad import NetVLAD
os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'

class Flatten(nn.Module):
	def forward(self, input_data):
		return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
	def __init__(self, dim=1):
		super().__init__()
		self.dim = dim

	def forward(self, input_data):
		return F.normalize(input_data, p=2, dim=self.dim)


def get_pca_encoding(model, vlad_encoding):
	pca_encoding = model.WPCA(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
	return pca_encoding

def get_integral_feature(feat_in):
	"""
	Input/Output as [N,D,H,W] where N is batch size and D is descriptor dimensions
	For VLAD, D = K x d where K is the number of clusters and d is the original descriptor dimensions
	"""
	feat_out = torch.cumsum(feat_in, dim=-1)
	feat_out = torch.cumsum(feat_out, dim=-2)
	feat_out = torch.nn.functional.pad(feat_out, (1, 0, 1, 0), "constant", 0)
	return feat_out


def get_square_regions_from_integral(feat_integral, patch_size, patch_stride):
	"""
	Input as [N,D,H+1,W+1] where additional 1s for last two axes are zero paddings
	regSize and regStride are single values as only square regions are implemented currently
	"""
	N, D, H, W = feat_integral.shape

	if feat_integral.get_device() == -1:
		conv_weight = torch.ones(D, 1, 2, 2)
	else:
		conv_weight = torch.ones(D, 1, 2, 2, device=feat_integral.get_device())
	conv_weight[:, :, 0, -1] = -1
	conv_weight[:, :, -1, 0] = -1
	feat_regions = torch.nn.functional.conv2d(feat_integral, conv_weight, stride=patch_stride, groups=D, dilation=patch_size)
	return feat_regions / (patch_size ** 2)


def get_vgg16():
	enc_dim = 512
	enc = models.vgg16()
	layers = list(enc.features.children())[:-2]
	# only train conv5_1, conv5_2, and conv5_3 (leave rest same as Imagenet trained weights)
	for layer in layers[:-5]:
		for p in layer.parameters():
			p.requires_grad = False
	enc = nn.Sequential(*layers)
	return enc, enc_dim
	

def get_netvlad(encoder, encoder_dim, config):
	# config['global_params'] is passed as config
	nn_model = nn.Module()
	nn_model.add_module('encoder', encoder)

	if config.pooling.lower() == 'netvlad':
		net_vlad = NetVLAD(num_clusters=int(config.num_clusters), dim=encoder_dim, vladv2=config.vladv2)
		nn_model.add_module('pool', net_vlad)
	elif config.pooling.lower() == 'patchnetvlad':
		net_vlad = PatchNetVLAD(num_clusters=int(config.num_clusters), dim=encoder_dim, vladv2=config.vladv2, patch_sizes=config.patch_sizes, strides=config.strides)
		nn_model.add_module('pool', net_vlad)
	elif config.pooling.lower() == 'max':
		global_pool = nn.AdaptiveMaxPool2d((1, 1))
		nn_model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
	elif config.pooling.lower() == 'avg':
		global_pool = nn.AdaptiveAvgPool2d((1, 1))
		nn_model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
	else:
		raise ValueError('Unknown pooling type: ' + config.pooling)

	if config.append_pca_layer:
		num_pcs = int(config.num_pcs)
		netvlad_output_dim = encoder_dim
		if config.pooling.lower() in ['netvlad', 'patchnetvlad']:
			netvlad_output_dim *= int(config.num_clusters)

		pca_conv = nn.Conv2d(netvlad_output_dim, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
		nn_model.add_module('WPCA', nn.Sequential(*[pca_conv, Flatten(), L2Norm(dim=-1)]))

	return nn_model

#
# class NetVLAD(nn.Module):
# 	"""NetVLAD layer implementation"""
#
# 	def __init__(self, num_clusters=64, dim=128,
# 				 normalize_input=True, vladv2=False, use_faiss=True):
# 		"""
#		 Args:
#			 num_clusters : int
#				 The number of clusters
#			 dim : int
#				 Dimension of descriptors
#			 normalize_input : bool
#				 If true, descriptor-wise L2 normalization is applied to input.
#			 vladv2 : bool
#				 If true, use vladv2 otherwise use vladv1
#		 """
# 		super().__init__()
# 		self.num_clusters = num_clusters
# 		self.dim = dim
# 		self.alpha = 0
# 		self.vladv2 = vladv2
# 		self.normalize_input = normalize_input
# 		self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
# 		# noinspection PyArgumentList
# 		self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
# 		self.use_faiss = use_faiss
#
# 	def init_params(self, clsts, traindescs):
# 		if not self.vladv2:
# 			clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
# 			dots = np.dot(clstsAssign, traindescs.T)
# 			dots.sort(0)
# 			dots = dots[::-1, :]  # sort, descending
#
# 			self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
# 			# noinspection PyArgumentList
# 			self.centroids = nn.Parameter(torch.from_numpy(clsts))
# 			# noinspection PyArgumentList
# 			self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
# 			self.conv.bias = None
# 		else:
# 			if not self.use_faiss:
# 				knn = NearestNeighbors(n_jobs=-1)
# 				knn.fit(traindescs)
# 				del traindescs
# 				ds_sq = np.square(knn.kneighbors(clsts, 2)[1])
# 				del knn
# 			else:
# 				index = faiss.IndexFlatL2(traindescs.shape[1])
# 				# noinspection PyArgumentList
# 				index.add(traindescs)
# 				del traindescs
# 				# noinspection PyArgumentList
# 				ds_sq = np.square(index.search(clsts, 2)[1])
# 				del index
#
# 			self.alpha = (-np.log(0.01) / np.mean(ds_sq[:, 1] - ds_sq[:, 0])).item()
# 			# noinspection PyArgumentList
# 			self.centroids = nn.Parameter(torch.from_numpy(clsts))
# 			del clsts, ds_sq
#
# 			# noinspection PyArgumentList
# 			self.conv.weight = nn.Parameter(
# 				(2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
# 			)
# 			# noinspection PyArgumentList
# 			self.conv.bias = nn.Parameter(
# 				- self.alpha * self.centroids.norm(dim=1)
# 			)
#
# 	def forward(self, x):
# 		N, D = x.shape[:2]
# 		if self.normalize_input:
# 			x = F.normalize(x, p=2, dim=1)  # across descriptor dim
# 		# soft-assignment
# 		soft_assign = self.conv(x).view(N, self.num_clusters, -1)
# 		soft_assign = F.softmax(soft_assign, dim=1)
# 		x_flatten = x.view(N, D, -1)
# 		# calculate residuals to each clusters
# 		vlad = torch.zeros([N, self.num_clusters, D], dtype=x.dtype, layout=x.layout, device=x.device)
# 		for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
# 			residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
# 					   self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
# 			residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
# 			vlad[:, C:C + 1, :] = residual.sum(dim=-1)
#
# 		vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
# 		vlad = vlad.view(x.size(0), -1)  # flatten
# 		vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
# 		return vlad

class PatchNetVLAD(nn.Module):
	"""NetVLAD layer implementation"""

	def __init__(self, num_clusters=64, dim=128, normalize_input=True, vladv2=False, use_faiss=True, patch_sizes='4', strides='1'):
		"""
		Args:
			num_clusters : int
				The number of clusters
			dim : int
				Dimension of descriptors
			normalize_input : bool
				If true, descriptor-wise L2 normalization is applied to input.
			vladv2 : bool
				If true, use vladv2 otherwise use vladv1
			use_faiss: bool
				Default true, if false don't use faiss for similarity search
			patch_sizes: string
				comma separated string of patch sizes
			strides: string
				comma separated string of strides (for patch aggregation)
		"""
		super().__init__()
		self.num_clusters = num_clusters
		self.dim = dim
		self.alpha = 0
		self.vladv2 = vladv2
		self.normalize_input = normalize_input
		self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
		# noinspection PyArgumentList
		self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
		self.use_faiss = use_faiss
		self.padding_size = 0
		patch_sizes = patch_sizes.split(",")
		strides = strides.split(",")
		self.patch_sizes = []
		self.strides = []
		for patch_size, stride in zip(patch_sizes, strides):
			self.patch_sizes.append(int(patch_size))
			self.strides.append(int(stride))

	def init_params(self, clsts, traindescs):
		if not self.vladv2:
			clsts_assign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
			dots = np.dot(clsts_assign, traindescs.T)
			dots.sort(0)
			dots = dots[::-1, :]  # sort, descending

			self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
			# noinspection PyArgumentList
			self.centroids = nn.Parameter(torch.from_numpy(clsts))
			# noinspection PyArgumentList
			self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clsts_assign).unsqueeze(2).unsqueeze(3))
			self.conv.bias = None
		else:
			if not self.use_faiss:
				knn = NearestNeighbors(n_jobs=-1)
				knn.fit(traindescs)
				del traindescs
				ds_sq = np.square(knn.kneighbors(clsts, 2)[1])
				del knn
			else:
				index = faiss.IndexFlatL2(traindescs.shape[1])
				# noinspection PyArgumentList
				index.add(traindescs)
				del traindescs
				# noinspection PyArgumentList
				ds_sq = index.search(clsts, 2)[1]
				del index

			self.alpha = (-np.log(0.01) / np.mean(ds_sq[:, 1] - ds_sq[:, 0])).item()
			# noinspection PyArgumentList
			self.centroids = nn.Parameter(torch.from_numpy(clsts))
			del clsts, ds_sq

			# noinspection PyArgumentList
			self.conv.weight = nn.Parameter(
				(2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
			)
			# noinspection PyArgumentList
			self.conv.bias = nn.Parameter(
				- self.alpha * self.centroids.norm(dim=1)
			)

	def forward(self, x):
		N, C, H, W = x.shape

		if self.normalize_input:
			x = F.normalize(x, p=2, dim=1)  # across descriptor dim

		# soft-assignment
		soft_assign = self.conv(x).view(N, self.num_clusters, H, W)
		soft_assign = F.softmax(soft_assign, dim=1)

		# calculate residuals to each cluster
		store_residual = torch.zeros([N, self.num_clusters, C, H, W], dtype=x.dtype, layout=x.layout, device=x.device)
		for j in range(self.num_clusters):  # slower than non-looped, but lower memory usage
			residual = x.unsqueeze(0).permute(1, 0, 2, 3, 4) - \
				self.centroids[j:j + 1, :].expand(x.size(2), x.size(3), -1, -1).permute(2, 3, 0, 1).unsqueeze(0)

			residual *= soft_assign[:, j:j + 1, :].unsqueeze(2)  # residual should be size [N K C H W]
			store_residual[:, j:j + 1, :, :, :] = residual

		vlad_global = store_residual.view(N, self.num_clusters, C, -1)
		vlad_global = vlad_global.sum(dim=-1)
		store_residual = store_residual.view(N, -1, H, W)


		vladflattened = []
		vlad_local = []
		if 0 not in self.strides:
			ivlad = get_integral_feature(store_residual)
			for patch_size, stride in zip(self.patch_sizes, self.strides):
				vladflattened.append(get_square_regions_from_integral(ivlad, int(patch_size), int(stride)))

			
			for thisvlad in vladflattened:  # looped to avoid GPU memory issues with certain config combinations
				thisvlad = thisvlad.view(N, self.num_clusters, C, -1)
				thisvlad = F.normalize(thisvlad, p=2, dim=2)
				thisvlad = thisvlad.view(x.size(0), -1, thisvlad.size(3))
				thisvlad = F.normalize(thisvlad, p=2, dim=1)
				vlad_local.append(thisvlad)
		else:
			vlad_local = None

		vlad_global = F.normalize(vlad_global, p=2, dim=2)
		vlad_global = vlad_global.view(x.size(0), -1)
		vlad_global = F.normalize(vlad_global, p=2, dim=1)

		return vlad_local, vlad_global  # vlad_local is a list of tensors
		

"""============================================================="""
class Network(torch.nn.Module):
	def __init__(self, opt):
		super(Network, self).__init__()

		self.pars  = opt
		self.name = opt.arch
		vgg, vgg_dim = get_vgg16()
		self.netvlad = get_netvlad(vgg, vgg_dim, opt)
		
		pretrained = not opt.not_pretrained
		if pretrained:
			print(f'load netvlad ckpt: {opt.vlad_ckpt}')
			checkpoint = torch.load(opt.vlad_ckpt, map_location=lambda storage, loc: storage)
			self.netvlad.load_state_dict(checkpoint['state_dict'], strict=False)

		if 'frozen' in opt.arch:
			for module in self.model.modules():
				print(module)
				module.eval()
				module.train = lambda _: None

		#self.head = torch.nn.Sequential(
		#torch.nn.Linear(32768, 4096),
		#torch.nn.ReLU(),
		#torch.nn.Linear(4096, opt.embed_dim)
		#)

	def forward(self, x):
	
		image_encoding = self.netvlad.encoder(x)
		vlad_local, vlad_global = self.netvlad.pool(image_encoding)
		patches, enc_out = vlad_local, vlad_global
		#B,D,N = patches.size()
		#patches = patches[0].view(B,D,int(N**0.5), int(N**0.5))
		patches = None
		enc_out = enc_out.view(enc_out.size(0),-1)
		
		#x = self.head(enc_out)
		x = get_pca_encoding(self.netvlad, enc_out)

		if 'normalize' in self.pars.arch:
			x = torch.nn.functional.normalize(x, dim=-1)
		
		return x, (enc_out, patches)
