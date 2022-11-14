import architectures.resnet50
import architectures.googlenet
import architectures.bninception
import architectures.resnet50_diml
import architectures.vit
import architectures.patchnetvlad
import architectures.swin
import architectures.cvt
import architectures.cvt_cross 
import architectures.densenet
def select(arch, opt):

		
	if arch.lower().startswith('resnet50_diml'):
		return resnet50_diml.Network(opt)
	if arch.lower().startswith('resnet50'):
		return resnet50.Network(opt)
	if  arch.lower().startswith('irresnet50'):
		return resnet50.IRResnet(opt)
	if  arch.lower().startswith('densenet'):
		return densenet.Network(opt)

	if arch.startswith('vit'):
		return vit.Network(opt)

	if 'netvlad' in arch:
		return patchnetvlad.Network(opt)
	if 'swin' in arch:
		return swin.Network(opt)
	if arch.lower().startswith('cvt'):
		if 'diml' in arch: # structural loss
			return cvt.DIML(opt)
		print(opt.arch)
		return cvt.Network(opt)
	

