"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import re
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _load_state_dict(model, state_dict):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)

"""============================================================="""
class Network(torch.nn.Module):
	def __init__(self, opt):
		super(Network, self).__init__()

		self.pars  = opt
		self.name = opt.arch
		if '201' in opt.arch:
			self.last_in = 1920
			self.model = models.densenet201(pretrained=False)
			if not opt.not_pretrained:
				state_dict = torch.load('/home/czhang/Pretrained_models/DenseNet/densenet201-c1103571.pth')
				_load_state_dict(self.model, state_dict)
		elif '169' in opt.arch:
			self.last_in = 1664
			self.model = models.densenet169(pretrained=False)
			if not opt.not_pretrained:
				state_dict = torch.load('/home/czhang/Pretrained_models/DenseNet/densenet169-b2777c0a.pth')
				_load_state_dict(self.model, state_dict)

		if 'frozen' in opt.arch:
			for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
				module.eval()
				module.train = lambda _: None

		del self.model.classifier
		self.model.last_linear = torch.nn.Linear(self.last_in, opt.embed_dim)


	def forward(self, x, **kwargs):

		features = self.model.features(x)
		features = F.relu(features, inplace=True)

		no_avg_feat = features
		x = F.adaptive_avg_pool2d(features, (1,1))
		x = torch.flatten(x, 1)
		enc_out = x
		x = self.model.last_linear(x)

		if 'normalize' in self.pars.arch:
			x = torch.nn.functional.normalize(x, dim=-1)


		return x, (enc_out, no_avg_feat)
