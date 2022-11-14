#!/usr/bin/env python

'''
This code trains the CVT on MSLS using distillation, from pretrained netvlad.

'''
from __future__ import print_function
import logging
import argparse
import configparser
import os
import sys
import random
import shutil
from os.path import join, isfile
from os import makedirs
from datetime import datetime
import time

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
import numpy as np

from training_tools.train_epoch import train_epoch
from training_tools.val import val
from training_tools.tools import save_checkpoint
from datasets.msls import input_transform, MSLS
import architectures as archs
import parameters as par
from tqdm.auto import trange
from utilities import logger


if __name__ == "__main__":

	################### INPUT ARGUMENTS ###################
	parser = argparse.ArgumentParser(description='MSLS-CVT-train-distill')

	parser = par.basic_training_parameters(parser)
	parser = par.batch_creation_parameters(parser)
	parser = par.batchmining_specific_parameters(parser)
	parser = par.loss_specific_parameters(parser)

	##### Read in parameters
	opt = parser.parse_args()
	print(opt)
	opt.source_path += '/' + opt.dataset
	opt.save_path += '/' + opt.dataset
	print(opt.save_path)
	opt.savename = opt.group + '_s{}'.format(opt.seed)

	writer = SummaryWriter(
		log_dir=join(opt.save_path, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.group))
	# write checkpoints in logdir
	logdir = writer.file_writer.get_logdir()
	opt.save_path = logdir
	file_handler = logging.FileHandler(filename=os.path.join(logdir, 'log.txt'))
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

	sub_loggers = ['Train', 'Test', 'Model Grad']
	LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=False)

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	device = opt.device = torch.device('cuda')

	torch.backends.cudnn.deterministic = True;
	np.random.seed(opt.seed);
	random.seed(opt.seed)
	torch.manual_seed(opt.seed);
	torch.cuda.manual_seed(opt.seed);
	torch.cuda.manual_seed_all(opt.seed)

	optimizer = None
	scheduler = None
	

	print('===> Building model')

	teacher_model = archs.select('netvlad_pca128', opt)
	model = archs.select(opt.arch, opt)
		
	teacher_model.eval()
	
	if opt.resume_path:
		checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage)
		model.load_state_dict(checkpoint['state_dict'])
		opt.start_epoch = checkpoint['epoch']

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		opt.is_parallel=True

	if opt.optim.lower() == 'adam':
		optimizer = optim.Adam(filter(lambda par: par.requires_grad,
									  model.parameters()), lr=opt.lr)  # , betas=(0,0.9))
	elif opt.optim.lower() == 'sgd':
		optimizer = optim.SGD(filter(lambda par: par.requires_grad,
									 model.parameters()), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightdecay)

		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrstep,
											  gamma=opt.lrgamma)
	else:
		raise ValueError('Unknown optimizer: ' + opt.optim)

	criterion = nn.TripletMarginLoss(margin=opt.margin ** 0.5, p=2, reduction='sum').to(device)

	model = model.to(device)
	
	if opt.distill or opt.rand_distill:
		teacher_model = teacher_model.to(device)
	else:
		teacher_model = None
	
	
	if opt.resume_path:
		optimizer.load_state_dict(checkpoint['optimizer'])

	print('===> Loading dataset(s)')
	train_cities = 'london' if opt.mini_data else ''
	val_cities = 'cph' if opt.mini_data else ''
	trainval_cities = 'amman'

	train_dataset = MSLS(opt.source_path, mode='train',
						 nNeg=opt.nNeg, cities=train_cities,
						 transform=input_transform(resize=(opt.imageresizeh, opt.imageresizew)),
						 bs=opt.cachebatchsize,
						 threads=opt.kernels,
						 margin=opt.margin,
						 exclude_panos=True,
						 distill=opt.distill, 
						 rand_distill=opt.rand_distill)

	validation_dataset = MSLS(opt.source_path, mode='val', cities=val_cities,
							  transform=input_transform(resize=(opt.imageresizeh, opt.imageresizew)),
							  bs=opt.cachebatchsize,
							  threads=opt.kernels,
							  margin=opt.margin,
							  posDistThr=25)
							  
	trainval_dataset = MSLS(opt.source_path, mode='val', cities=trainval_cities,
							  transform=input_transform(resize=(opt.imageresizeh, opt.imageresizew)),
							  bs=opt.cachebatchsize,
							  threads=opt.kernels,
							  margin=opt.margin,
							  posDistThr=25)

	print('===> Training query set:', len(train_dataset.qIdx))
	print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))
	print('===> Evaluating on trainval set, query count:', len(trainval_dataset.qIdx))
	print('===> Training model')

	opt.save_file_path = join(logdir, 'checkpoints')
	makedirs(opt.save_file_path)

	not_improved = 0
	best_score = 0
	best_recalls = None
	if opt.resume_path:
		not_improved = checkpoint['not_improved']
		best_score = checkpoint['best_score']

	for epoch in trange(opt.start_epoch, opt.n_epochs, desc='Epoch number'.rjust(15), position=0):

		logging_logger.info(f"Epoch {epoch} start")
		start = time.time()
		
		train_dataset.new_epoch()
		
		if False:
			if opt.distill:
				if epoch==0:
					train_dataset.generate_triplets(teacher_model, opt.embed_dim)
				train_dataset.set_triplets_increase_sub()
				
			elif opt.rand_distill:
				if epoch==0:
					train_dataset.generate_postive_negative_candidates(teacher_model, opt.embed_dim)
					
			else:
				tqdm.write('====> Building Cache')
				train_dataset.update_subcache(teacher_model, opt.embed_dim)
			
			
		train_loss = train_epoch(train_dataset, model, optimizer, criterion, device, epoch, opt, logging_logger, LOG, teacher_model)

		LOG.progress_saver['Train'].log('epochs', epoch)
		result_metrics = {'loss': np.array(train_loss)}

		for metricname, metricval in result_metrics.items():
			LOG.progress_saver['Train'].log(metricname, metricval)
		LOG.progress_saver['Train'].log('time', np.round(time.time() - start, 4))

		logging_logger.info(f"Average Loss:  {train_loss:.4f}")

		if scheduler is not None:
			scheduler.step()

		if (epoch+1) % opt.evalevery == 0:
			recalls = val(validation_dataset, model, device, opt, writer, epoch, is_train=False, pbar_position=1, trunc_nums=[0])
			
			trainval_recalls = val(trainval_dataset, model, device, opt, writer, epoch, is_train=True, pbar_position=1, trunc_nums=[0])

		
			for k, v in recalls.items():
				LOG.progress_saver['Test'].log(k, v)

			is_best = recalls['globalR_5'] > best_score
			if is_best:
				not_improved = 0
				best_score = recalls['globalR_5']
				best_recalls = recalls
			else:
				not_improved += 1
			logging_logger.info(f'Saving epoch {epoch} checkpoint ...')
			save_checkpoint({
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'recalls': recalls,
				'best_score': best_score,
				'not_improved': not_improved,
				'optimizer': optimizer.state_dict(),
				'parallel': opt.is_parallel,
			}, opt, is_best)

			print('###########')
			for key, value in trainval_recalls.items():
				logging_logger.info(f'TrainVal {key}: {value:.4f}')
			for key, value in recalls.items():
				logging_logger.info(f'Now {key}: {value:.4f}')
			for key, value in best_recalls.items():
				logging_logger.info(f'Best {key}: {value:.4f}')

			if opt.patience > 0 and not_improved > opt.patience:
				logging_logger.info(f'Performance did not improve for {opt.patience*opt.evalevery} epochs. Stopping.')
				break

		LOG.update(all=True)



	print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
	writer.close()

	torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
	# memory after runs

	print('Done')