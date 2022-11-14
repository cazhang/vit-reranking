"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import comet_ml
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm, trange
from utilities.misc import load_checkpoint
import torch.nn.functional as F
import shutil

import parameters as par
from evaluation.eval_attn_diml import evaluate, evaluate_patch_similarity

"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)

##### Read in parameters
opt = parser.parse_args()


### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler   as dsamplers
import datasets	  as datasets
import criteria	  as criteria
import batchminer	as bmine
import evaluation	as eval
from utilities import misc
from utilities import logger


"""==================================================================================================="""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset
print(opt.save_path)

#Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

opt.pretrained = not opt.not_pretrained

"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"

"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)

"""==================================================================================================="""
##################### NETWORK SETUP ##################
opt.device = torch.device('cuda')

"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders = {}
datasets	= datasets.select(opt.dataset, opt, opt.source_path)

dataloaders['testing']	= torch.utils.data.DataLoader(datasets['testing'],	num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
opt.n_classes  = len(dataloaders['testing'].dataset.avail_classes)
model	  = archs.select(opt.arch, opt)
_  = model.to(opt.device)

"""============================================================================"""
################### Summary #########################3
data_text  = 'Dataset:\t {}'.format(opt.dataset.upper())
setup_text = 'Objective:\t {}'.format(opt.loss.upper())
arch_text  = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
summary	= data_text+'\n'+setup_text+'\n'+arch_text
print(summary)

"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
loss_args  = {'batch': None, 'labels':None, 'batch_features':None, 'f_embed':None}


# prepare path
CUB_LOGS = {
	'Margin_b06_deit_small': ['CUB_Margin_b06_Distance_vit_small_s0']
}

CARS_LOGS = {

}

SOP_LOGS = {
}

if opt.dataset == 'cub200':
	LOGS = CUB_LOGS
elif opt.dataset == 'cars196':
	LOGS = CARS_LOGS
else:
	LOGS = SOP_LOGS

		
results = []
methods = []
data = {
	k: []
	for k in ['method', 'r1', 'rp', 'mapr']
}

trunc_nums = [0, 100]

for method, info in LOGS.items():
	path = f'Training_Results/{opt.dataset}/{info[0]}/best.pth' 
	best_metrics = load_checkpoint(model, None, path)
	print(best_metrics)
	
	if MODE_DEBUG: # use finetune version
		sims = evaluate_patch_similarity(model.model,  datasets['testing'], dataloaders['testing'])
		exit(1)
	
	result = evaluate(model, datasets['testing'], dataloaders['testing'], opt.training, trunc_nums,
					  use_uniform=opt.use_uniform, grid_size=opt.grid_size, use_inverse=opt.use_inverse,
					  temperature=opt.temperature, use_cls_token=opt.use_cls_token,
					  attn_blk_ind=opt.blk_ind, use_ot=opt.use_ot, ot_part=opt.ot_part,
					  to_submit=opt.to_submit, use_minus=opt.use_minus,
					  use_rollout=opt.use_rollout, use_soft=opt.use_soft)

	print(result)
	result['method'] = [f'{method} + ours ({trunc}) + part {opt.ot_part}' for trunc in trunc_nums]
	for k, v in data.items():
		v.extend(result[k])

df = pd.DataFrame(data)
os.makedirs('test_results', exist_ok=True)
output_path = f'test_results/test_diml_{opt.dataset}.csv'
df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
