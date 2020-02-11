import os, sys
import numpy as np
import torch
from .networks import HourglassNet
from .BaseModel import BaseModel

from torch.autograd import Variable
from .losses import FocalLoss
from .losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from .utils import _sigmoid

from .decode import ctdet_decode

class Hourglass(BaseModel):
	def name(self):
		return 'Hourglass'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.opt = opt
		self.model = HourglassNet(opt.heads, 2)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)
		#if self.opt.load_model != '':
		#	self.load_model(self.opt.load_model, self.opt.resume, self.opt.lr, self.opt.lr_step)

		if len(self.opt.gpu_ids) > 1:
			# use multiple gpus
			self.model = torch.nn.DataParallel(self.model, device_ids=self.opt.gpu_ids).to(opt.device)
		elif len(self.opt.gpu_ids) > 0:
			# only use a single gpu
			self.model = self.model.to(opt.device)

		self.crit = FocalLoss()
		self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
			RegLoss() if opt.reg_loss == 'sl1' else None
		self.crit_wh = NormRegL1loss() if opt.norm_wh else self.crit_reg

		self.num_stacks = 2

	def set_input(self, input, mode='train'):
		if mode == 'train' or mode == 'val':
		# input is a dict
			self.imgs = Variable(input['img'].to(self.opt.device))
			self.hm = Variable(input['hm'].to(self.opt.device))
			self.reg_mask = Variable(input['reg_mask'].to(self.opt.device))
			self.ind = Variable(input['ind'].to(self.opt.device))
			self.wh = Variable(input['wh'].to(self.opt.device))
			self.reg = Variable(input['reg'].to(self.opt.device))
			if self.opt.stack:
				self.shm = Variable(input['shm'].to(self.opt.device))
		else:
			self.imgs = Variable(input['img'].to(self.opt.device))
		#print(self.reg_mask)	

	def get_current_loss(self):
		return self.outputs[-1], self.loss, self.loss_stats

	def forward(self):
		#self.model.train()
		self.outputs = self.model(self.imgs)
		self.loss, self.loss_stats = self.compute_loss(self.outputs)
		return self.outputs[-1], self.loss, self.loss_stats

	def compute_loss(self, outptus):
		hm_loss, wh_loss, off_loss = 0, 0, 0
		if self.opt.stack:
			shm_loss = 0
		for s in range(self.num_stacks):
			output = self.outputs[s]
			output['hm'] = _sigmoid(output['hm'])
			hm_loss += self.crit(output['hm'], self.hm) / self.num_stacks

			if self.opt.stack:
				output['shm'] = _sigmoid(output['shm'])
				shm_loss += self.crit(output['shm'], self.shm) / self.num_stacks
			if self.opt.wh_weight > 0:
				wh_loss += self.crit_wh(output['wh'], self.reg_mask, self.ind, self.wh) / self.num_stacks

			if self.opt.reg_offset and self.opt.off_weight > 0:
				off_loss += self.crit_reg(output['reg'], self.reg_mask, self.ind, self.reg) / self.num_stacks


		loss = self.opt.hm_weight * hm_loss + self.opt.wh_weight * wh_loss + self.opt.off_weight * off_loss
		if self.opt.stack:
			loss += self.opt.shm_weight * shm_loss
		#loss = hm_loss
		loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss}
		if self.opt.stack:
			loss_stats['shm_loss'] = shm_loss
		return loss, loss_stats

	def inference(self):
		self.model.eval()
		#print(self.imgs.size())
		outputs = self.model(self.imgs)
		#hm = outputs['hm'].sigmoid_()
		#wh = outputs['wh']
		#reg = outputs['reg']
		#dets = ctdet_decode(hm, wh, reg=reg)
		return outputs[-1]

	def backward(self):
		#loss = self.loss.mean()
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()
			

	def optimize_parameters(self):
		self.forward()
		self.backward()
	
	def save(self, name, epoch):
		path = os.path.join(self.opt.checkpoint_dir, name, self.opt.model+'_'+str(epoch) + '.pth')
		torch.save(self.model.state_dict(), path)
		# path = os.path.join(self.opt.checkpoint_dir, name, name+"_"+str(epoch)+'.pth')
		# print(path)
		# if isinstance(self.model, torch.nn.DataParallel):
		# 	state_dict = self.model.module.state_dict()
		# else:
		# 	state_dict = self.model.state_dict()
		# data = {'epoch': epoch,
		# 		'state_dict': state_dict}
		# if not (self.optimizer is None):
		# 	data['optimizer'] = self.optimizer.state_dict()
		# torch.save(data, path)

	def load(self, model_path, resume=False, lr=None, lr_step=None):
		if len(self.opt.gpu_ids) > 1:
			state_dict = torch.load(model_path)
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				if 'module' in k:
					name = k
					new_state_dict[name] = v
				else:
					name = 'module.' + k
					new_state_dict[name] = v
			self.model.load_state_dict(new_state_dict)
		else:
			state_dict = torch.load(model_path)
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in state_dict.items():
				if 'module.' in k:
					name = k[7:] # remove module.
					new_state_dict[name] = v
				else:
					new_state_dict[k] = v
			self.model.load_state_dict(new_state_dict)

		# start_epoch = 0
		# checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
		# print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
		# state_dict = checkpoint['state_dict']
		# state_dict = {}

		# # convert data_parallel to model
		# for k in state_dict:
		# 	if k.startswith('module') and not k.startswith('module_list'):
		# 		state_dict[k[7:]] = state_dict_[k]
		# 	else:
		# 		state_dict[k] = state_dict_[k]
		# model_state_dict = self.model.state_dict()

		# # check loaded parameters and created mdoel parameters
		# for k in state_dict:
		# 	if k in model_state_dict:
		# 		if state_dict[k].shape != model_state_dict[k].shape:
		# 			print('skip loading parameters, the model is not fully load')
		# 			state_dict[k] = model_state_dict[k]
		# for k in model_state_dict:
		# 	if not (k in state_dict):
		# 		print('the model is not fully load')
		# 		state_dict[k] = model_state_dict[k]

		# self.model.load_state_dict(state_dict, strict=False)

		# # resume optimizer parameters
		# if self.optimizer is not None and resume:
		# 	if 'optimizer' in cehckpoint:
		# 		self.optimizer.load_state_dict(checkpoint['optimizer'])
		# 		state_epoch = checkpoint['epoch']
		# 		start_lr = lr
		# 		for step in lr_step:
		# 			if start_epoch >= step:
		# 				start_lr *= 0.1
		# 		for param_group in self.optimizer.param_groups:
		# 			param_group['lr'] = start_lr
		# 	else:
		# 		print('no optimizer parameters in checkpoint')
