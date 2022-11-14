"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import comet_ml
import warnings
warnings.filterwarnings("ignore")
import logging
import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from evaluation.metrics import get_metrics, get_metrics_rank
from tqdm import tqdm, trange
import shutil

import parameters	as par
import torch.nn.functional as F
from utilities.diml import Sinkhorn, calc_similarity


"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)

##### Read in parameters
opt = parser.parse_args()


"""==================================================================================================="""
opt.savename = opt.group + '_s{}'.format(opt.seed)


"""==================================================================================================="""
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
full_training_start_time = time.time()



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
model	  = archs.select(opt.arch, opt)

if opt.fc_lr<0:
	to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
else:
	all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'last_linear' not in x[0], model.named_parameters()))]
	fc_params		 = model.model.last_linear.parameters()
	to_optim		  = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
						 {'params':fc_params,'lr':opt.fc_lr,'weight_decay':opt.decay}]

_  = model.to(opt.device)




"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders = {}
datasets	= datasets.select(opt.dataset, opt, opt.source_path)

dataloaders['evaluation'] = torch.utils.data.DataLoader(datasets['evaluation'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
dataloaders['testing']	= torch.utils.data.DataLoader(datasets['testing'],	num_workers=opt.kernels, batch_size=200, shuffle=False)
if opt.use_tv_split:
	dataloaders['validation'] = torch.utils.data.DataLoader(datasets['validation'], num_workers=opt.kernels, batch_size=opt.bs,shuffle=False)

train_data_sampler	  = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict, datasets['training'].image_list)
if train_data_sampler.requires_storage:
	train_data_sampler.create_storage(dataloaders['evaluation'], model, opt.device)

dataloaders['training'] = torch.utils.data.DataLoader(datasets['training'], num_workers=opt.kernels, batch_sampler=train_data_sampler)

opt.n_classes  = len(dataloaders['training'].dataset.avail_classes)




"""============================================================================"""
#################### CREATE LOGGING FILES ###############
sub_loggers = ['Train', 'Test', 'Model Grad']
if opt.use_tv_split: sub_loggers.append('Val')
LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=False)





"""============================================================================"""
#################### LOSS SETUP ####################
batchminer   = bmine.select(opt.batch_mining, opt)
criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
_ = criterion.to(opt.device)

if 'criterion' in train_data_sampler.name:
	train_data_sampler.internal_criterion = criterion




"""============================================================================"""
#################### OPTIM SETUP ####################
if opt.optim == 'adam':
	optimizer	= torch.optim.Adam(to_optim)
elif opt.optim == 'sgd':
	optimizer	= torch.optim.SGD(to_optim, momentum=0.9)
else:
	raise Exception('Optimizer <{}> not available!'.format(opt.optim))
scheduler	= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)


"""============================================================================"""
#################### METRIC COMPUTER ####################
opt.rho_spectrum_embed_dim = opt.embed_dim


"""============================================================================"""
################### Summary #########################3
data_text  = 'Dataset:\t {}'.format(opt.dataset.upper())
setup_text = 'Objective:\t {}'.format(opt.loss.upper())
miner_text = 'Batchminer:\t {}'.format(opt.batch_mining if criterion.REQUIRES_BATCHMINER else 'N/A')
arch_text  = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
summary	= data_text+'\n'+setup_text+'\n'+miner_text+'\n'+arch_text
print(summary)


"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
loss_args  = {'batch':None, 'labels':None, 'batch_features':None, 'f_embed':None}

best_r1 = 0
best_rp = 0
best_mapr = 0
patience = 0

file_handler = logging.FileHandler(filename=os.path.join(opt.save_path, 'log.txt'))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

for handler in logging.root.handlers[:]:
	logging.root.removeHandler(handler)

logging.basicConfig(
	level=logging.INFO, 
	format='[%(asctime)s] %(levelname)s - %(message)s',
	handlers=handlers,
)

logging_logger = logging.getLogger('root')
logging_logger.setLevel(logging.INFO)

if torch.cuda.device_count() > 1:
	model = nn.DataParallel(model)

if opt.resume_path:
	best_metrics, start_epoch = misc.load_checkpoint(model, optimizer, opt.resume_path)
	print(best_metrics)
	best_r1 = best_metrics['r1']
	best_rp = best_metrics['rp']
	best_mapr = best_metrics['mapr']
else:
	start_epoch = 0



for epoch in range(start_epoch, opt.n_epochs):
	epoch_start_time = time.time()

	if epoch>0 and opt.data_idx_full_prec and train_data_sampler.requires_storage:
		train_data_sampler.full_storage_update(dataloaders['evaluation'], model, opt.device)

	opt.epoch = epoch
	### Scheduling Changes specifically for cosine scheduling
	if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

	"""======================================="""
	if train_data_sampler.requires_storage:
		train_data_sampler.precompute_indices()


	"""======================================="""
	### Train one epoch
	start = time.time()
	_ = model.train()

	loss_collect = []
	data_iterator = tqdm(dataloaders['training'], desc='Epoch {} Training...'.format(epoch))
	logging_logger.info(f"Epoch {epoch} start")
	print(opt.save_path)

	for i,out in enumerate(data_iterator):
		class_labels, input, input_indices = out
		### Compute Embedding
		input	  = input.to(opt.device)
		model_args = {'x':input.to(opt.device)}
		# Needed for MixManifold settings.
		if 'mix' in opt.arch: model_args['labels'] = class_labels
		embeds  = model(**model_args)
	
		if isinstance(embeds, tuple): 
			embeds, (global_enc, features) = embeds
		### Compute Loss
		loss_args['batch']		  = embeds
		loss_args['labels']		 = class_labels
		# loss_args['f_embed']		= model.module.model.last_linear
		loss_args['class_token'] = global_enc
		loss	  = criterion(**loss_args)

		optimizer.zero_grad()

		if loss is not None:
			loss.backward()

			# torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

			### Compute Model Gradients and log them!
			grads			  = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
			grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
			LOG.progress_saver['Model Grad'].log('Grad L2',  grad_l2,  group='L2')
			LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

			### Update network weights!
			optimizer.step()
			loss_collect.append(loss.item())

		###
		iter_count += 1

		if i==len(dataloaders['training'])-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))

		"""======================================="""
		if train_data_sampler.requires_storage and train_data_sampler.update_storage:
			train_data_sampler.replace_storage_entries(embeds.detach().cpu(), input_indices)


	result_metrics = {'loss': np.mean(loss_collect)}

	####
	LOG.progress_saver['Train'].log('epochs', epoch)
	for metricname, metricval in result_metrics.items():
		LOG.progress_saver['Train'].log(metricname, metricval)
	LOG.progress_saver['Train'].log('time', np.round(time.time()-start, 4))

	"""======================================="""
	### Evaluate Metric for Training & Test (& Validation)
	if (epoch+1) % opt.evalevery == 0:
		_ = model.eval()
		print('\nComputing Testing Metrics...')
		# eval.evaluate(opt.dataset, LOG, metric_computer, [dataloaders['testing']],	model, opt, opt.evaltypes, opt.device, log_key='Test')
		###
		dataloader = dataloaders['testing']

		with torch.no_grad():
			target_labels = []
			feature_bank = []
			feature_bank_center = []
			labels = []
			final_iter = tqdm(dataloader, desc='Embedding Data...'.format(1))
			image_paths = [x[0] for x in dataloader.dataset.image_list]
			for idx, inp in enumerate(final_iter):
				input_img, target = inp[1], inp[0]
				target_labels.extend(target.numpy().tolist())
				out = model(input_img.to(opt.device))
				if isinstance(out, tuple): out, (global_enc, features) = out
				feature_bank.append(out.data)
				feature_bank_center.append(global_enc.data)
				labels.append(target)

			feature_bank = torch.cat(feature_bank, dim=0)
			feature_bank_center = torch.cat(feature_bank_center, dim=0)
			labels = torch.cat(labels, dim=0)

			N, C, H, W = feature_bank.size()
			feature_bank = feature_bank.view(N, C, -1)

			feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1)
			feature_bank_center = torch.nn.functional.normalize(feature_bank_center, p=2, dim=1)

			overall_r1 = 0.0
			overall_rp = 0.0
			overall_mapr = 0.0

			approx_num = 100

			for idx in trange(len(feature_bank)):
				anchor_center = feature_bank_center[idx]
				approx_sim, _ = calc_similarity(None, anchor_center, None, feature_bank_center, 0)
				approx_sim[idx] = -100
				approx_tops = torch.argsort(approx_sim, descending=True)
				top_idx = approx_tops[0:approx_num]
				trucated_feature_bank = feature_bank[top_idx]

				anchor = feature_bank[idx]
				sim, _ = calc_similarity(anchor, anchor_center, feature_bank[top_idx], feature_bank_center[top_idx], 1, opt.use_uniform, use_inverse=opt.use_inverse, temperature=opt.temperature, use_cls_token=opt.use_cls_token, use_minus=opt.use_minus)

				approx_sim = approx_sim[top_idx]
				assert len(approx_sim) == len(sim)
				rank_in_tops = torch.argsort(sim + approx_sim, descending=True)
				rank_in_tops_real = top_idx[rank_in_tops]
				final_tops = torch.cat([rank_in_tops_real, approx_tops[approx_num:]], dim=0)
				# sim[idx] = -100
				r1, rp, mapr = get_metrics_rank(final_tops.data.cpu(), labels[idx], labels)
				overall_r1 += r1
				overall_rp += rp
				overall_mapr += mapr


			overall_r1 = overall_r1 / float(N / 100)
			overall_rp = overall_rp / float(N / 100)
			overall_mapr = overall_mapr / float(N / 100)

			is_best = False
			if overall_r1 > best_r1:
				best_r1 = overall_r1
				is_best = True
				patience = 0
			else:
				patience+=1
			if overall_rp > best_rp:
				best_rp = overall_rp
			if overall_mapr > best_mapr:
				best_mapr = overall_mapr

			all_metrics = {
				'r1': overall_r1,
				'rp': overall_rp,
				'mapr': overall_mapr,
			}
			best_metrics = {
				'r1': best_r1,
				'rp': best_rp,
				'mapr': best_mapr,
			}

			for k, v in all_metrics.items():
				LOG.progress_saver['Test'].log(k, v)

			logging_logger.info('saving checkpoint...')
			misc.save_checkpoint(model, optimizer, os.path.join(opt.save_path, 'latest.pth'), all_metrics, best_metrics, epoch)
			if is_best:
				logging_logger.info('saving best checkpoint...')
				shutil.copy2(os.path.join(opt.save_path, 'latest.pth'), os.path.join(opt.save_path, 'best.pth'))


			print('###########')
			logging_logger.info('Now rank-1 acc=%f, RP=%f, MAP@R=%f' % (overall_r1, overall_rp, overall_mapr))
			logging_logger.info('Best rank-1 acc=%f, RP=%f, MAP@R=%f' % (best_r1,  best_rp, best_mapr))

			if patience > opt.max_patience:
				logging_logger.info(f'Not improving for {opt.max_patience*opt.evalevery} epochs, terminating.')
				break


	LOG.update(all=True)

	"""======================================="""
	### Learning Rate Scheduling Step
	if opt.scheduler != 'none':
		scheduler.step()

	print('Total Epoch Runtime: {0:4.2f}s'.format(time.time()-epoch_start_time))
	print('\n-----\n')




"""======================================================="""
### CREATE A SUMMARY TEXT FILE
summary_text = ''
full_training_time = time.time()-full_training_start_time
summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time/60,2))

summary_text += '---------------\n'
for sub_logger in LOG.sub_loggers:
	metrics	   = LOG.graph_writer[sub_logger].ov_title
	summary_text += '{} metrics: {}\n'.format(sub_logger.upper(), metrics)

with open(opt.save_path+'/training_summary.txt','w') as summary_file:
	summary_file.write(summary_text)
