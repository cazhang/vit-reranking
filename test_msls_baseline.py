#!/usr/bin/env python

'''
This code tests the CVT on MSLS.

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

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
import numpy as np

from training_tools.train_epoch import train_epoch
from training_tools.val import val as validate
from training_tools.tools import save_checkpoint
from datasets.msls import input_transform, MSLS
import architectures as archs
import parameters as par
from tqdm.auto import trange


if __name__ == "__main__":

	################### INPUT ARGUMENTS ###################
	parser = argparse.ArgumentParser(description='MSLS-test')

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
	breakpoint()
	model = archs.select(opt.arch, opt)

	if opt.resume_path:
		print(f'===> Loding ckpt: {opt.resume_path}')
		checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage)
		
		state_dict = checkpoint['state_dict']
		new_state_dict = {}
		for key, val in state_dict.items():
			if 'module' in key: 
				new_key = key.replace('module.', '')
				new_state_dict[new_key] = val 
			else:
				new_state_dict[key] = val 
				
		print(new_state_dict.keys())
		model.load_state_dict(new_state_dict)
		opt.start_epoch = checkpoint['epoch']

	isParallel = False
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		isParallel=True


	model = model.to(device)

	# print('===> Loading training dataset(s)')
	# train_dataset = MSLS(opt.source_path, mode='train',
	#					  nNeg=opt.nNeg, cities='trondheim,london,boston',
	#					  transform=input_transform(resize=(opt.imageresizeh, opt.imageresizew)),
	#					  bs=opt.cachebatchsize,
	#					  threads=opt.kernels,
	#					  margin=opt.margin,
	#					  exclude_panos=True)
	#print('===> Training query set:', len(train_dataset.qIdx))

	print('===> Loading validation dataset(s)')
	validation_dataset = MSLS(opt.source_path, mode='val', cities='',
							  transform=input_transform(resize=(opt.imageresizeh, opt.imageresizew)),
							  bs=opt.cachebatchsize,
							  threads=opt.kernels,
							  margin=opt.margin,
							  posDistThr=25)


	print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))
	print('===> Evaluating model')

	trunc_nums = [0, 100]

	recalls = validate(validation_dataset, model, device, opt, writer, epoch_num=0, is_train=False, pbar_position=1, trunc_nums=trunc_nums)

	print('###########')
	for key, value in recalls.items():
		logging_logger.info(f'Now {key}: {value:.4f}')

	torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
	# memory after runs

	print('Done')
