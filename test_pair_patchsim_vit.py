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
import datasampler as dsamplers
import datasets as datasets
import criteria as criteria
import batchminer as bmine
import evaluation as eval
from utilities import misc
from utilities import logger

"""==================================================================================================="""
opt.source_path += '/' + opt.dataset
opt.save_path += '/' + opt.dataset
print(opt.save_path)

# Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs % opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

opt.pretrained = not opt.not_pretrained

"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic = True;
np.random.seed(opt.seed);
random.seed(opt.seed)
torch.manual_seed(opt.seed);
torch.cuda.manual_seed(opt.seed);
torch.cuda.manual_seed_all(opt.seed)

"""==================================================================================================="""
##################### NETWORK SETUP ##################
opt.device = torch.device('cuda')

"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders = {}
datasets = datasets.select(opt.dataset, opt, opt.source_path)

dataloaders['testing'] = torch.utils.data.DataLoader(datasets['testing'], num_workers=opt.kernels, batch_size=opt.bs,
													 shuffle=False)
opt.n_classes = len(dataloaders['testing'].dataset.avail_classes)
model = archs.select(opt.arch, opt)
_ = model.to(opt.device)

"""============================================================================"""
################### Summary #########################3
data_text = 'Dataset:\t {}'.format(opt.dataset.upper())
setup_text = 'Objective:\t {}'.format(opt.loss.upper())
arch_text = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
summary = data_text + '\n' + setup_text + '\n' + arch_text
print(summary)

"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
loss_args = {'batch': None, 'labels': None, 'batch_features': None, 'f_embed': None}

# prepare path
CUB_LOGS = {
	#'Margin_b06_vit': ['CUB_Margin_b06_Distance_s0_vit'],
	'Margin_b06_deit_small': ['CUB_Margin_b06_Distance_vit_small_s0']
}

CARS_LOGS = {
	# 'Margin_b06_res50': ['CARS_Margin_b06_Distance_s0_resnet50'],
	'pvlad_normalize': ['CARS_Margin_b06_Distance_pvlad_s0_vlad']
	# 'Margin_b06_vit_v2': ['CARS_Margin_b06_Distance_s0_6'],

}

SOP_LOGS = {
	'Margin_b06_res50': ['CUB_Margin_b06_Distance_s0_res50'],
}

if opt.dataset == 'cub200':
	LOGS = CUB_LOGS
elif opt.dataset == 'cars196':
	LOGS = CARS_LOGS
else:
	LOGS = SOP_LOGS

from proofs.eval_pair_patch_sim import evaluate, evaluate_patch_similarity

results = []
methods = []
data = {
	k: []
	for k in ['method', 'r1', 'rp', 'mapr']
}

MODE_DEBUG = True
MODE_PRE = False
if MODE_DEBUG and MODE_PRE:  # pretrained version
	del model
	from architectures import resnet50

	model = resnet50.Network(opt)
	model.to(opt.device)

	# result = evaluate_patch_similarity(model, datasets['testing'], dataloaders['testing'])
	result = evaluate(model, datasets['testing'], dataloaders['testing'], True, 0, use_uniform=False, grid_size=7)

	exit(1)

trunc_nums = [0]

for method, info in LOGS.items():
	path = f'Training_Results/{opt.dataset}/{info[0]}/best.pth'
	best_metrics = load_checkpoint(model, None, path)
	print(best_metrics)

	if MODE_DEBUG:  # finetuned version
		result = evaluate(model, datasets['testing'], dataloaders['testing'], opt.training,
						  trunc_nums=trunc_nums, use_uniform=opt.use_uniform,
						  grid_size=opt.grid_size, blk_ind=0,
						  to_submit=opt.to_submit,
						  use_relu=opt.use_relu)

		#result = evaluate_patch_similarity(model, datasets['testing'], dataloaders['testing'], layer_id=layer_id)
		exit(1)

