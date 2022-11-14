"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch
from torch import einsum
import torch.nn as nn
import timm
from einops import rearrange, repeat

from utilities.cross import PreNorm
"""============================================================="""

class Network(nn.Module):
	def __init__(self, opt):
		super(Network, self).__init__()
		self.pars = opt
		self.name = 'swin_tiny_patch4_window7_224'
		self.last_in = 768
		self.model = timm.create_model(self.name, pretrained=False)

		if self.name == 'swin_tiny_patch4_window7_224':
			checkpoint_url = '/home/czhang/Pretrained_models/Transformers/swin_tiny_patch4_window7_224.pth'
		state_dict = torch.load(checkpoint_url)['model']
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

	def forward(self, x, **kwargs):

		x = self.model.patch_embed(x)
		if self.model.absolute_pos_embed is not None:
			x = x + self.absolute_pos_embed
		x = self.model.pos_drop(x)
		x = self.model.layers(x)
		x = self.model.norm(x)  # B L C
		# breakpoint()
		no_avg_feat = x
		
		x = self.model.avgpool(x.transpose(1, 2))  # B C 1
		x = torch.flatten(x, 1)
		enc_out = x 
		x = self.model.head(x)
		if 'normalize' in self.pars.arch:
			x = torch.nn.functional.normalize(x, dim=-1)
		
		return x, (enc_out, no_avg_feat)
		
'''apply cross attention to exchange info'''
class CrossAttention(nn.Module):
	def __init__(self, dim, heads = 12, dim_head = 64, dropout = 0.):
		super().__init__()
		inner_dim = dim_head *  heads
		project_out = not (heads == 1 and dim_head == dim)

		self.heads = heads
		self.scale = dim_head ** -0.5

		self.to_k = nn.Linear(dim, inner_dim , bias = True)
		self.to_v = nn.Linear(dim, inner_dim , bias = True)
		self.to_q = nn.Linear(dim, inner_dim, bias = True)

		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, dim),
			nn.Dropout(dropout)
		) if project_out else nn.Identity()

	def forward(self, x_qkv):
		b, n, _, h = *x_qkv.shape, self.heads

		k = self.to_k(x_qkv)
		k = rearrange(k, 'b n (h d) -> b h n d', h = h)

		v = self.to_v(x_qkv)
		v = rearrange(v, 'b n (h d) -> b h n d', h = h)

		q = self.to_q(x_qkv[:, 0].unsqueeze(1))
		q = rearrange(q, 'b n (h d) -> b h n d', h = h)


		dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

		attn = dots.softmax(dim=-1)

		out = einsum('b h i j, b h j d -> b h i d', attn, v)
		out = rearrange(out, 'b h n d -> b n (h d)')
		out =  self.to_out(out)
		return out

class CrossNet_vit(nn.Module):
	def __init__(self, opt):
		super(CrossNet_vit, self).__init__()
		self.pars = opt
		self.dim = 768
		self.head = torch.nn.Linear(self.dim, opt.embed_dim)
		self.cross_attention = nn.ModuleList([])
		for _ in range(opt.cross_attn_depth):
			self.cross_attention.append(nn.ModuleList([
				PreNorm(self.dim, CrossAttention(dim=self.dim, heads=12, dim_head=64, dropout=0.2)),
				PreNorm(self.dim, CrossAttention(dim=self.dim, heads=12, dim_head=64, dropout=0.2)),
			]))
	
	def forward(self, xs, ys):
		''' x and y are vit tokens
		Args: x_class: b x 768
			x_patch: b x 196 x 768
		Return:
			x_class: b x 128
			y_class: b x 128
		'''
		for cross_attn_src, cross_attn_tgt in self.cross_attention:
			x_class, x_patch = xs[:, 0], xs[:, 1::]
			y_class, y_patch = ys[:, 0], ys[:, 1::]

			# cross attn for anchor
			cal_q = x_class.unsqueeze(1)
			cal_qkv = torch.cat((cal_q, y_patch), dim=1)
			cal_out = cal_q + cross_attn_src(cal_qkv)
			xs = torch.cat((cal_out, x_patch), dim=1)

			# cross attn for other
			cal_q = y_class.unsqueeze(1)
			cal_qkv = torch.cat((cal_q, x_patch), dim=1)
			cal_out = cal_q + cross_attn_tgt(cal_qkv)
			ys = torch.cat((cal_out, y_patch), dim=1)
		
		x_class = self.head(xs[:,0])
		y_class = self.head(ys[:,0])
		
		if 'normalize' in self.pars.arch:
			x_class = torch.nn.functional.normalize(x_class, dim=-1)
			y_class = torch.nn.functional.normalize(y_class, dim=-1)
		return x_class, y_class
		
		
		
		
		
	

