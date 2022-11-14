from functools import partial
from itertools import repeat

import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

print(f'Torch_major: {TORCH_MAJOR}, Torch_minor: {TORCH_MINOR}')
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'

# From PyTorch internals
def _ntuple(n):
	def parse(x):
		if isinstance(x, container_abcs.Iterable):
			return x
		return tuple(repeat(x, n))

	return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
	"""Subclass torch's LayerNorm to handle fp16."""

	def forward(self, x: torch.Tensor):
		orig_type = x.dtype
		ret = super().forward(x.type(torch.float32))
		return ret.type(orig_type)


class QuickGELU(nn.Module):
	def forward(self, x: torch.Tensor):
		return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
	def __init__(self,
				 in_features,
				 hidden_features=None,
				 out_features=None,
				 act_layer=nn.GELU,
				 drop=0.):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x


class Attention(nn.Module):
	def __init__(self,
				 dim_in,
				 dim_out,
				 num_heads,
				 qkv_bias=False,
				 attn_drop=0.,
				 proj_drop=0.,
				 method='dw_bn',
				 kernel_size=3,
				 stride_kv=1,
				 stride_q=1,
				 padding_kv=1,
				 padding_q=1,
				 with_cls_token=True,
				 **kwargs
				 ):
		super().__init__()
		self.stride_kv = stride_kv
		self.stride_q = stride_q
		self.dim = dim_out
		self.num_heads = num_heads
		# head_dim = self.qkv_dim // num_heads
		self.scale = dim_out ** -0.5
		self.with_cls_token = with_cls_token

		self.ret_attn = kwargs['ret_attn']

		self.conv_proj_q = self._build_projection(
			dim_in, dim_out, kernel_size, padding_q,
			stride_q, 'linear' if method == 'avg' else method
		)
		self.conv_proj_k = self._build_projection(
			dim_in, dim_out, kernel_size, padding_kv,
			stride_kv, method
		)
		self.conv_proj_v = self._build_projection(
			dim_in, dim_out, kernel_size, padding_kv,
			stride_kv, method
		)

		self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
		self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
		self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim_out, dim_out)
		self.proj_drop = nn.Dropout(proj_drop)

	def _build_projection(self,
						  dim_in,
						  dim_out,
						  kernel_size,
						  padding,
						  stride,
						  method):
		if method == 'dw_bn':
			proj = nn.Sequential(OrderedDict([
				('conv', nn.Conv2d(
					dim_in,
					dim_in,
					kernel_size=kernel_size,
					padding=padding,
					stride=stride,
					bias=False,
					groups=dim_in
				)),
				('bn', nn.BatchNorm2d(dim_in)),
				('rearrage', Rearrange('b c h w -> b (h w) c')),
			]))
		elif method == 'avg':
			proj = nn.Sequential(OrderedDict([
				('avg', nn.AvgPool2d(
					kernel_size=kernel_size,
					padding=padding,
					stride=stride,
					ceil_mode=True
				)),
				('rearrage', Rearrange('b c h w -> b (h w) c')),
			]))
		elif method == 'linear':
			proj = None
		else:
			raise ValueError('Unknown method ({})'.format(method))

		return proj

	def forward_conv(self, x, h, w):
		if self.with_cls_token:
			cls_token, x = torch.split(x, [1, h*w], 1)

		x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

		if self.conv_proj_q is not None:
			q = self.conv_proj_q(x)
		else:
			q = rearrange(x, 'b c h w -> b (h w) c')

		if self.conv_proj_k is not None:
			k = self.conv_proj_k(x)
		else:
			k = rearrange(x, 'b c h w -> b (h w) c')

		if self.conv_proj_v is not None:
			v = self.conv_proj_v(x)
		else:
			v = rearrange(x, 'b c h w -> b (h w) c')

		if self.with_cls_token:
			q = torch.cat((cls_token, q), dim=1)
			k = torch.cat((cls_token, k), dim=1)
			v = torch.cat((cls_token, v), dim=1)

		return q, k, v

	def forward(self, x, h, w):
		if (
				self.conv_proj_q is not None
				or self.conv_proj_k is not None
				or self.conv_proj_v is not None
		):
			q, k, v = self.forward_conv(x, h, w)

		q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
		k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
		v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
		

		attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
		attn = F.softmax(attn_score, dim=-1)
		attn = self.attn_drop(attn)
		weights = attn if self.ret_attn else None
		x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
		x = rearrange(x, 'b h t d -> b t (h d)')

		x = self.proj(x)
		x = self.proj_drop(x)

		return x, weights

	@staticmethod
	def compute_macs(module, input, output):
		# T: num_token
		# S: num_token
		input = input[0]
		flops = 0

		_, T, C = input.shape
		H = W = int(np.sqrt(T-1)) if module.with_cls_token else int(np.sqrt(T))

		H_Q = H / module.stride_q
		W_Q = H / module.stride_q
		T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

		H_KV = H / module.stride_kv
		W_KV = W / module.stride_kv
		T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

		# C = module.dim
		# S = T
		# Scaled-dot-product macs
		# [B x T x C] x [B x C x T] --> [B x T x S]
		# multiplication-addition is counted as 1 because operations can be fused
		flops += T_Q * T_KV * module.dim
		# [B x T x S] x [B x S x C] --> [B x T x C]
		flops += T_Q * module.dim * T_KV

		if (
				hasattr(module, 'conv_proj_q')
				and hasattr(module.conv_proj_q, 'conv')
		):
			params = sum(
				[
					p.numel()
					for p in module.conv_proj_q.conv.parameters()
				]
			)
			flops += params * H_Q * W_Q

		if (
				hasattr(module, 'conv_proj_k')
				and hasattr(module.conv_proj_k, 'conv')
		):
			params = sum(
				[
					p.numel()
					for p in module.conv_proj_k.conv.parameters()
				]
			)
			flops += params * H_KV * W_KV

		if (
				hasattr(module, 'conv_proj_v')
				and hasattr(module.conv_proj_v, 'conv')
		):
			params = sum(
				[
					p.numel()
					for p in module.conv_proj_v.conv.parameters()
				]
			)
			flops += params * H_KV * W_KV

		params = sum([p.numel() for p in module.proj_q.parameters()])
		flops += params * T_Q
		params = sum([p.numel() for p in module.proj_k.parameters()])
		flops += params * T_KV
		params = sum([p.numel() for p in module.proj_v.parameters()])
		flops += params * T_KV
		params = sum([p.numel() for p in module.proj.parameters()])
		flops += params * T

		module.__flops__ += flops


class Block(nn.Module):

	def __init__(self,
				 dim_in,
				 dim_out,
				 num_heads,
				 mlp_ratio=4.,
				 qkv_bias=False,
				 drop=0.,
				 attn_drop=0.,
				 drop_path=0.,
				 act_layer=nn.GELU,
				 norm_layer=nn.LayerNorm,
				 **kwargs):
		super().__init__()

		self.with_cls_token = kwargs['with_cls_token']

		self.norm1 = norm_layer(dim_in)
		self.attn = Attention(
			dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
			**kwargs
		)

		self.drop_path = DropPath(drop_path) \
			if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim_out)

		dim_mlp_hidden = int(dim_out * mlp_ratio)
		self.mlp = Mlp(
			in_features=dim_out,
			hidden_features=dim_mlp_hidden,
			act_layer=act_layer,
			drop=drop
		)


	def forward(self, x, h, w):
		res = x

		x = self.norm1(x)
		attn, weights = self.attn(x, h, w)
		self._probs.append(weights)
	
		x = res + self.drop_path(attn)
		x = x + self.drop_path(self.mlp(self.norm2(x)))

		return x

class ConvEmbed(nn.Module):
	""" Image to Conv Embedding

	"""

	def __init__(self,
				 patch_size=7,
				 in_chans=3,
				 embed_dim=64,
				 stride=4,
				 padding=2,
				 norm_layer=None):
		super().__init__()
		patch_size = to_2tuple(patch_size)
		self.patch_size = patch_size

		self.proj = nn.Conv2d(
			in_chans, embed_dim,
			kernel_size=patch_size,
			stride=stride,
			padding=padding
		)
		self.norm = norm_layer(embed_dim) if norm_layer else None

	def forward(self, x):
		x = self.proj(x)

		B, C, H, W = x.shape
		x = rearrange(x, 'b c h w -> b (h w) c')
		if self.norm:
			x = self.norm(x)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

		return x


class VisionTransformer(nn.Module):
	""" Vision Transformer with support for patch or hybrid CNN input stage
	"""
	def __init__(self,
				 patch_size=16,
				 patch_stride=16,
				 patch_padding=0,
				 in_chans=3,
				 embed_dim=768,
				 depth=12,
				 num_heads=12,
				 mlp_ratio=4.,
				 qkv_bias=False,
				 drop_rate=0.,
				 attn_drop_rate=0.,
				 drop_path_rate=0.,
				 act_layer=nn.GELU,
				 norm_layer=nn.LayerNorm,
				 init='trunc_norm',
				 **kwargs):
		super().__init__()
		self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

		self.rearrage = None

		self.depth = depth
		self.patch_embed = ConvEmbed(
			# img_size=img_size,
			patch_size=patch_size,
			in_chans=in_chans,
			stride=patch_stride,
			padding=patch_padding,
			embed_dim=embed_dim,
			norm_layer=norm_layer
		)

		with_cls_token = kwargs['with_cls_token']
		if with_cls_token:
			self.cls_token = nn.Parameter(
				torch.zeros(1, 1, embed_dim)
			)
		else:
			self.cls_token = None

		self.pos_drop = nn.Dropout(p=drop_rate)
		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

		blocks = []
		for j in range(depth):
			blocks.append(
				Block(
					dim_in=embed_dim,
					dim_out=embed_dim,
					num_heads=num_heads,
					mlp_ratio=mlp_ratio,
					qkv_bias=qkv_bias,
					drop=drop_rate,
					attn_drop=attn_drop_rate,
					drop_path=dpr[j],
					act_layer=act_layer,
					norm_layer=norm_layer,
					**kwargs
				)
			)
		self.blocks = nn.ModuleList(blocks)

		if self.cls_token is not None:
			trunc_normal_(self.cls_token, std=.02)

		if init == 'xavier':
			self.apply(self._init_weights_xavier)
		else:
			self.apply(self._init_weights_trunc_normal)

	def _init_weights_trunc_normal(self, m):
		if isinstance(m, nn.Linear):
			#logging.info('=> init weight of Linear from trunc norm')
			trunc_normal_(m.weight, std=0.02)
			if m.bias is not None:
				#logging.info('=> init bias of Linear to zeros')
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def _init_weights_xavier(self, m):
		if isinstance(m, nn.Linear):
			#logging.info('=> init weight of Linear from xavier uniform')
			nn.init.xavier_uniform_(m.weight)
			if m.bias is not None:
				#logging.info('=> init bias of Linear to zeros')
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def forward(self, x):
		x = self.patch_embed(x)
		B, C, H, W = x.size()

		x = rearrange(x, 'b c h w -> b (h w) c')

		cls_tokens = None
		if self.cls_token is not None:
			# stole cls_tokens impl from Phil Wang, thanks
			cls_tokens = self.cls_token.expand(B, -1, -1)
			x = torch.cat((cls_tokens, x), dim=1)

		x = self.pos_drop(x)

		for i, blk in enumerate(self.blocks):
			blk._probs = []
			x = blk(x, H, W)

		if self.cls_token is not None:
			cls_tokens, x = torch.split(x, [1, H*W], 1)
		x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

		return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Module):
	def __init__(self,
				 in_chans=3,
				 num_classes=1000,
				 act_layer=nn.GELU,
				 norm_layer=nn.LayerNorm,
				 init='trunc_norm',
				 spec=None):
		super().__init__()
		self.spec = spec
		self.num_classes = num_classes
		self.num_stages = spec['NUM_STAGES']
		for i in range(self.num_stages):
			kwargs = {
				'patch_size': spec['PATCH_SIZE'][i],
				'patch_stride': spec['PATCH_STRIDE'][i],
				'patch_padding': spec['PATCH_PADDING'][i],
				'embed_dim': spec['DIM_EMBED'][i],
				'depth': spec['DEPTH'][i],
				'num_heads': spec['NUM_HEADS'][i],
				'mlp_ratio': spec['MLP_RATIO'][i],
				'qkv_bias': spec['QKV_BIAS'][i],
				'drop_rate': spec['DROP_RATE'][i],
				'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
				'drop_path_rate': spec['DROP_PATH_RATE'][i],
				'with_cls_token': spec['CLS_TOKEN'][i],
				'method': spec['QKV_PROJ_METHOD'][i],
				'kernel_size': spec['KERNEL_QKV'][i],
				'padding_q': spec['PADDING_Q'][i],
				'padding_kv': spec['PADDING_KV'][i],
				'stride_kv': spec['STRIDE_KV'][i],
				'stride_q': spec['STRIDE_Q'][i],
				'ret_attn': spec['RET_ATTN'][i],
			}

			stage = VisionTransformer(
				in_chans=in_chans,
				init=init,
				act_layer=act_layer,
				norm_layer=norm_layer,
				**kwargs
			)
			setattr(self, f'stage{i}', stage)

			in_chans = spec['DIM_EMBED'][i]

		dim_embed = spec['DIM_EMBED'][-1]
		self.norm = norm_layer(dim_embed)
		self.cls_token = spec['CLS_TOKEN'][-1]

		# Classifier head
		self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()

	def init_weights(self, pretrained='', pretrained_layers=[], verbose=False):
		if os.path.isfile(pretrained):
			pretrained_dict = torch.load(pretrained, map_location='cpu')
			logging.info(f'=> loading pretrained model {pretrained}')
			model_dict = self.state_dict()
			pretrained_dict = {
				k: v for k, v in pretrained_dict.items()
				if k in model_dict.keys()
			}
			need_init_state_dict = {}
			for k, v in pretrained_dict.items():
				need_init = (
						k.split('.')[0] in pretrained_layers
						or pretrained_layers[0] is '*'
				)
				if need_init:
					if verbose:
						logging.info(f'=> init {k} from {pretrained}')
					if 'pos_embed' in k and v.size() != model_dict[k].size():
						size_pretrained = v.size()
						size_new = model_dict[k].size()
						logging.info(
							'=> load_pretrained: resized variant: {} to {}'
								.format(size_pretrained, size_new)
						)

						ntok_new = size_new[1]
						ntok_new -= 1

						posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

						gs_old = int(np.sqrt(len(posemb_grid)))
						gs_new = int(np.sqrt(ntok_new))

						logging.info(
							'=> load_pretrained: grid-size from {} to {}'
								.format(gs_old, gs_new)
						)

						posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
						zoom = (gs_new / gs_old, gs_new / gs_old, 1)
						posemb_grid = scipy.ndimage.zoom(
							posemb_grid, zoom, order=1
						)
						posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
						v = torch.tensor(
							np.concatenate([posemb_tok, posemb_grid], axis=1)
						)

					need_init_state_dict[k] = v
			self.load_state_dict(need_init_state_dict, strict=False)

	@torch.jit.ignore
	def no_weight_decay(self):
		layers = set()
		for i in range(self.num_stages):
			layers.add(f'stage{i}.pos_embed')
			layers.add(f'stage{i}.cls_token')

		return layers

	def forward_features(self, x):
		for i in range(self.num_stages):
			x, cls_tokens = getattr(self, f'stage{i}')(x)

		if self.cls_token:
			x = self.norm(cls_tokens)
			x = torch.squeeze(x)
		else:
			x = rearrange(x, 'b c h w -> b (h w) c')
			x = self.norm(x)
			x = torch.mean(x, dim=1)

		return x

	def forward(self, x):
		x = self.forward_features(x)
		x = self.head(x)

		return x

	def both_forward(self, x):

		for i in range(self.num_stages):
			x, cls_tokens = getattr(self, f'stage{i}')(x)
		return x, cls_tokens

	def list_forward(self, x):
		out = []
		for i in range(self.num_stages):
			x, cls_tokens = getattr(self, f'stage{i}')(x)
			out.append(x)
		return out, cls_tokens


def get_cvt_spec(use_attn=False):
	msvit_spec = dict(INIT='trunc_norm',
					  NUM_STAGES=3,
					  PATCH_SIZE=[7, 3, 3],
					  PATCH_STRIDE=[4, 2, 2],
					  PATCH_PADDING=[2, 1, 1],
					  DIM_EMBED=[64, 192, 384],
					  NUM_HEADS=[1, 3, 6],
					  DEPTH=[1, 2, 10],
					  MLP_RATIO=[4.0, 4.0, 4.0],
					  ATTN_DROP_RATE=[0.0, 0.0, 0.0],
					  DROP_RATE=[0.0, 0.0, 0.0],
					  DROP_PATH_RATE=[0.0, 0.0, 0.1],
					  QKV_BIAS=[True, True, True],
					  CLS_TOKEN=[False, False, True],
					  POS_EMBED=[False, False, False],
					  QKV_PROJ_METHOD=['dw_bn', 'dw_bn', 'dw_bn'],
					  KERNEL_QKV=[3, 3, 3],
					  PADDING_KV=[1, 1, 1],
					  STRIDE_KV=[2, 2, 2],
					  PADDING_Q=[1, 1, 1],
					  STRIDE_Q=[1, 1, 1],
					  RET_ATTN=[use_attn, use_attn, use_attn],
					  )
	return msvit_spec


class Network(nn.Module):
	def __init__(self, opt):
		super(Network, self).__init__()
		self.pars = opt
		self.name = 'cvt-13-224x224'
		self.last_in = 384
		self.num_classes = opt.num_classes
		msvit_spec = get_cvt_spec(use_attn=opt.use_rollout)
		self.model = ConvolutionalVisionTransformer(
			in_chans=3,
			num_classes=self.num_classes,
			act_layer=QuickGELU,
			norm_layer=partial(LayerNorm, eps=1e-5),
			init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
			spec=msvit_spec
		)

		if self.name == 'cvt-13-224x224':
			checkpoint_url = '/home/czhang/Pretrained_models/Transformers/CvT-13-224x224-IN-1k.pth'
		state_dict = torch.load(checkpoint_url)
		to_delete_keys = []
		for key, val in state_dict.items():
			if 'head' in key:
				to_delete_keys.append(key)

		if opt.not_pretrained: # imagenet model
			if self.num_classes<1:
				for key in to_delete_keys:
					print('{} is deleted'.format(key))
					state_dict.pop(key)
				self.model.load_state_dict(state_dict, strict=False)
				print('Using Headless Imagenet pretrained model...')
			else:
				self.model.load_state_dict(state_dict, strict=True)
				print('Using Full Imagenet pretrained model...')
		else:
			for key in to_delete_keys:
				print('{} is deleted'.format(key))
				state_dict.pop(key)
			del self.model.head
			self.model.load_state_dict(state_dict, strict=False)
			self.model.head = torch.nn.Linear(self.last_in, opt.embed_dim)
			# init head
			#print('init head with trunc_normal...')
			trunc_normal_(self.model.head.weight, std=0.02)

		if 'frozen' in opt.arch:
			# for param in self.model.parameters():
			# 	param.requires_grad = False
			# freeze stage 1 and 2
			for name, child in (self.model.named_children()):
				if name.find('stage0') != -1 or name.find('stage1') != -1:
				#if name.find('stage0') != -1:
					print(f'freeze {name}')
					for param in child.parameters():
						param.requires_grad = False

	def forward(self, x, **kwargs):

		x, cls_token = self.model.both_forward(x)

		x = rearrange(x, 'b c h w -> b (h w) c')
		no_avg_feat = self.model.norm(x)

		x = self.model.norm(cls_token)
		x = torch.squeeze(x, dim=1)
		enc_out = x
		x = self.model.head(x)

		if 'normalize' in self.pars.arch:
			x = torch.nn.functional.normalize(x, dim=-1)
		return x, (enc_out, no_avg_feat)


class FPNetwork(nn.Module):
	def __init__(self, opt):
		super(FPNetwork, self).__init__()
		self.pars = opt
		self.name = 'cvt-13-224x224'
		self.last_in = 640
		msvit_spec = get_cvt_spec()
		self.model = ConvolutionalVisionTransformer(
			in_chans=3,
			num_classes=1000,
			act_layer=QuickGELU,
			norm_layer=partial(LayerNorm, eps=1e-5),
			init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
			spec=msvit_spec
		)

		if self.name == 'cvt-13-224x224':
			checkpoint_url = '/home/czhang/Pretrained_models/Transformers/CvT-13-224x224-IN-1k.pth'
		state_dict = torch.load(checkpoint_url)
		to_delete_keys = []
		for key, val in state_dict.items():
			if 'head' in key:
				print('{} is deleted'.format(key))
				to_delete_keys.append(key)

		if opt.not_pretrained:
			self.model.load_state_dict(state_dict, strict=True)
		else:
			for key in to_delete_keys:
				state_dict.pop(key)
			del self.model.head
			self.model.load_state_dict(state_dict, strict=False)
			self.model.head = torch.nn.Linear(self.last_in, opt.embed_dim)

		if 'frozen' in opt.arch:
			for param in self.model.parameters():
				param.requires_grad = False
		self.model.norms = []
		for i in range(msvit_spec['NUM_STAGES']):
			self.model.norms.append(nn.LayerNorm(msvit_spec['DIM_EMBED'][i]))
		self.resize = nn.AdaptiveAvgPool2d(7)
		self.gap = nn.AvgPool2d(7, stride=1)

	def forward(self, x, **kwargs):

		xs, cls_token = self.model.list_forward(x)
		ys = []

		for i, x in enumerate(xs):
			# print(x.size())
			#x = rearrange(x, 'b c h w -> b (h w) c')
			#x = self.model.norms[i](x)
			x = self.resize(x)
			if i==0:
				ys = x
			else:
				ys = torch.cat((ys,x), dim=1)

		ys = self.gap(ys).squeeze()

		ys = self.model.head(ys)

		if 'normalize' in self.pars.arch:
			ys = torch.nn.functional.normalize(ys, dim=-1)
		return ys, (None, None)
		

""" structural network """
class DIML(nn.Module):
	def __init__(self, opt):
		super(DIML, self).__init__()
		self.pars = opt
		self.name = 'cvt-13-224x224'
		self.last_in = 384
		self.grid_size = opt.grid_size
		msvit_spec = get_cvt_spec(use_attn = opt.use_rollout)
		self.model = ConvolutionalVisionTransformer(
			in_chans=3,
			num_classes=1000,
			act_layer=QuickGELU,
			norm_layer=partial(LayerNorm, eps=1e-5),
			init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
			spec=msvit_spec
		)

		if self.name == 'cvt-13-224x224':
			checkpoint_url = '/home/czhang/Pretrained_models/Transformers/CvT-13-224x224-IN-1k.pth'
		state_dict = torch.load(checkpoint_url)
		to_delete_keys = []
		for key, val in state_dict.items():
			if 'head' in key:
				print('{} is deleted'.format(key))
				to_delete_keys.append(key)

		if opt.not_pretrained:
			self.model.load_state_dict(state_dict, strict=True)
		else:
			for key in to_delete_keys:
				state_dict.pop(key)
			del self.model.head
			self.model.load_state_dict(state_dict, strict=False)
			self.model.last_linear = torch.nn.Conv2d(self.last_in, opt.embed_dim, 1)

		freeze_list = []
		names = []

		if 'noln' in opt.arch:
			for name, module in (self.model.named_modules()):
				names.append(name)
				if isinstance(module, nn.LayerNorm):
					freeze_list.append(name)
					module.weight.requires_grad = False
					module.bias.requires_grad = False

		if 'frozen' in opt.arch:
			#for param in self.model.parameters():
			#	param.requires_grad = False
			for name, child in (self.model.named_children()):
				names.append(name)
				if name.find('stage0') != -1 or name.find('stage1') != -1:
				#if name.find('stage0') != -1:
					for param in child.parameters():
						param.requires_grad = False

		self.down_sample = torch.nn.AdaptiveAvgPool2d(self.grid_size)

	def forward(self, x, **kwargs):
		#breakpoint()
		x, cls_token = self.model.both_forward(x)
		B,C,H,W = x.size()
		B,L,C = cls_token.size()

		x = rearrange(x, 'b c h w -> b (h w) c')
		x = self.model.norm(x)


		no_avg_feat = x.permute(0,2,1).contiguous().view(B,C,H,W)
		if no_avg_feat.size(-1)!=self.grid_size:
			no_avg_feat = self.down_sample(no_avg_feat)

		per_feat = self.model.last_linear(no_avg_feat)

		# missing norm to cls_token!!!
		cls_token = self.model.norm(cls_token)
		global_enc = cls_token.view(B, C, 1, 1)
		global_enc = self.model.last_linear(global_enc)
		global_enc = torch.flatten(global_enc, 1)
	
		return per_feat, (global_enc, no_avg_feat)
