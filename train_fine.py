from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import torch.utils.data
import numpy as np 

from opt.FineOptions import TrainOptions
from opt.FineTestOptions import TestOptions

from util.Visualizer import Visualizer
from util.evaluate import get_batch_statistics, ap_per_class
from data.CustomDataset import get_dataset

from models.get_model import create_model
from models.decode import ctdet_decode

def val(epoch):
	# validation can help us choose the best model
	# and prevent overfitting problems
	torch.backends.cudnn.benchmark = True
	opt = TestOptions().parse()
	vis = Visualizer(opt, env='validate')
	dataset = get_dataset(opt)
	valLoader = torch.utils.data.DataLoader(
		dataset,
		batch_size = 1, # only support batchsize = 1 for validation
		shuffle=False
	)
	model_path = os.path.join('checkpoints', opt.name, opt.model + '_' + str(epoch) + '.pth')
	print('load model from: ', model_path)
	val_model = create_model(opt)
	if len(opt.gpu_ids) > 1:
		val_model = val_model.module

	val_model.load(model_path)

	if opt.stack:
		avg_val_loss = {'loss': 0, 'hm_loss': 0, 'wh_loss': 0, 'off_loss': 0, 'shm_loss': 0}
	else:
		avg_val_loss = {'loss': 0, 'hm_loss': 0, 'wh_loss': 0, 'off_loss': 0}

	for i, inputs in enumerate(valLoader):
		val_model.set_input(inputs)
		with torch.no_grad():
			output, loss, loss_stat = val_model.forward()
		for k, v in loss_stat.items():
			avg_val_loss[k] += loss_stat[k]
		vis.plot_prediction(inputs['img'], output['hm'], output['wh'], output['reg'])
		vis.plot_heatmap(inputs['img'], inputs['hm'])
		vis.plot_heatmap(inputs['img'], output['hm'].cpu(), name='pred_hm')
	
	for k, v in avg_val_loss.items():
		avg_val_loss[k] /= len(dataset)

	print('In Epoch: %d for validation' % epoch)
	message = ''
	for k, v in avg_val_loss.items():
		message += k + ': %.4f ' % (v)
	print(message)
	return avg_val_loss

def train():
	torch.backends.cudnn.benchmark = True
	opt = TrainOptions().parse()
	vis = Visualizer(opt)
	dataset = get_dataset(opt)
	print(len(dataset))

	# this is for train
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size = opt.batch_size,
		shuffle=True
	)

	# create model here
	model = create_model(opt)

	if len(opt.gpu_ids) > 1:
		model = model.module

	if opt.load_model != '':
		model.load(opt.load_model, opt.resume, opt.lr, opt.lr_step)

	for epoch in range(opt.start_epochs, opt.start_epochs + opt.epochs):
		start_time = time.time()
		if opt.stack:
			avg_loss = {'loss': 0, 'hm_loss': 0, 'wh_loss': 0, 'off_loss': 0, 'shm_loss': 0}
		else:
			avg_loss = {'loss': 0, 'hm_loss': 0, 'wh_loss': 0, 'off_loss': 0}
		avg_val_loss = None
		count = 1

		for i, inputs in enumerate(dataloader):
			model.train()
			model.set_input(inputs)
			model.optimize_parameters()
			output, loss, loss_stat = model.get_current_loss()
			for k,v in loss_stat.items():
				avg_loss[k] += loss_stat[k]
			if i % opt.display_freq == 0:
				# the loss the the average of all the epochs between two display freqs
				for k,v in avg_loss.items():
					avg_loss[k] /= count
				count = 0
				# print losses
				vis.print_loss(avg_loss, epoch, i, len(dataloader))
				
				vis.plot_prediction(inputs['img'], output['hm'].clone(), output['wh'], output['reg'])
				vis.plot_heatmap(inputs['img'], inputs['hm'])
				vis.plot_heatmap(inputs['img'], output['hm'].cpu(), name='pred_hm')
				vis.plot_loss(avg_loss, epoch, i, len(dataloader))
				
				# plot current loss and outputs
				if opt.stack:
					avg_loss = {'loss': 0, 'hm_loss': 0, 'wh_loss': 0, 'off_loss': 0, 'shm_loss': 0}
				else:
					avg_loss = {'loss': 0, 'hm_loss': 0, 'wh_loss': 0, 'off_loss': 0}
				model.save(opt.name, 'latest')
			count += 1
			#break
		model.save(opt.name, epoch)
		
		if epoch % opt.val_freq == 0:
			# run validation set
			val_loss_stat = val(epoch)
			if avg_val_loss is None:
				avg_val_loss = val_loss_stat
				model.save(opt.name,'best')
			elif avg_val_loss['loss'] > val_loss_stat['loss']:
				avg_val_loss = val_loss_stat
				model.save(opt.name,'best')

if __name__ == '__main__':
	train()