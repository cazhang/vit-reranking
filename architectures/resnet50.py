"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.models as models

"""============================================================="""
class Network(torch.nn.Module):
	def __init__(self, opt):
		super(Network, self).__init__()

		self.pars  = opt
		self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)
		
		if not opt.not_pretrained:
			self.model.load_state_dict(torch.load('/home/czhang/Pretrained_models/ResNet/resnet50-19c8e357.pth'), strict=False)

		self.name = opt.arch

		if 'frozen' in opt.arch:
			for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
				module.eval()
				module.train = lambda _: None

		self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

		self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

		self.out_adjust = None


	def forward(self, x, **kwargs):
		x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
		for layerblock in self.layer_blocks:
			x = layerblock(x)
		no_avg_feat = x
		x = self.model.avgpool(x)
		enc_out = x = x.view(x.size(0),-1)

		x = self.model.last_linear(x)

		if 'normalize' in self.pars.arch:
			x = torch.nn.functional.normalize(x, dim=-1)
		if self.out_adjust and not self.train:
			x = self.out_adjust(x)

		return x, (enc_out, no_avg_feat)

"""============================================================="""

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative

class GeM(nn.Module):

	def __init__(self, p=3, eps=1e-6):
		super(GeM, self).__init__()
		self.p = Parameter(torch.ones(1) * p)
		self.eps = eps

	def forward(self, x):
		return gem(x, p=self.p, eps=self.eps)

	def __repr__(self):
		return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
			self.eps) + ')'

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

class L2N(nn.Module):

	def __init__(self, eps=1e-6):
		super(L2N, self).__init__()
		self.eps = eps

	def forward(self, x):
		return l2n(x, eps=self.eps)

	def __repr__(self):
		return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class IRResnet(torch.nn.Module):
	def __init__(self, opt):
		super(IRResnet, self).__init__()

		self.pars = opt
		self.name = opt.arch
		net_in = getattr(models, 'resnet50')(pretrained=False)

		if not opt.not_pretrained:
			net_in.load_state_dict(torch.load('/home/czhang/Pretrained_models/ResNet/resnet50-19c8e357.pth'),
									   strict=True)

		features = list(net_in.children())[:-2]
		self.features = nn.Sequential(*features)
		self.pool = GeM()
		self.norm = L2N()
		self.whiten = nn.Linear(2048, self.pars.embed_dim)

	def forward(self, x):

		feat = self.features(x)
		o = self.norm(self.pool(feat)).squeeze(-1).squeeze(-1)
		o = self.norm(self.whiten(o))
		return o, feat
