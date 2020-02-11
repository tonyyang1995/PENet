from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import torch.utils.data
import numpy as np 

from PIL import Image

from opt.CoarseTestOptions import TestOptions as CTestOptions
from opt.FineTestOptions import TestOptions

from util.Visualizer import Visualizer
from util.evaluate import post_processing, nms
from data.CustomDataset import get_dataset

from models.get_model import create_model
from models.decode import ctdet_decode

import torchvision.transforms as transforms

from PIL import Image
import time

def crop_eval():
	torch.backends.cudnn.benchmark = True
	opt = TestOptions().parse()
	vis = Visualizer(opt)

	dataset = get_dataset(opt)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size=1,
		shuffle=False
	)

	model = create_model(opt)
	model_path = os.path.join('checkpoints', opt.name, opt.model + '_best.pth')

	if len(opt.gpu_ids) > 1:
		model = model.module
	model.load(model_path)

	for i, inputs in enumerate(dataloader):
		imgs = inputs['imgs']
		offsets = inputs['offset']
		detections = None

		for idx, img in enumerate(imgs):
			offset = offsets[idx]
			left = int(offset[0])
			top = int(offset[1])

			inp = {"img": img}
			model.set_input(inp, mode='eval')
			with torch.no_grad():
				output = model.inference()

			dets = ctdet_decode(output['hm'].sigmoid_(), output['wh'], reg = output['reg'], K=100)
			dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

			detection = dets[0]
			detection[:, 0] = detection[:, 0] * 4 + left
			detection[:, 1] = detection[:, 1] * 4 + top
			detection[:, 2] = detection[:, 2] * 4 + left
			detection[:, 3] = detection[:, 3] * 4 + top

			detections = detection if detections is None else np.concatenate([detections, detection], axis=0)
		
		save_path = 'results/'+opt.name+'/cropval/'+inputs['img_name'][0]
		vis.save_preds(inputs['ori_img'], detections, save_path)
		save_dec_path = 'results/'+opt.name+'/cropvaldetection/'+inputs['img_name'][0]
		vis.save_dets(detections, save_dec_path)


def ori_eval():
	torch.backends.cudnn.benchmark = True
	opt = TestOptions().parse()
	vis = Visualizer(opt)

	dataset = get_dataset(opt)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size=1,
		shuffle=False
	)

	model = create_model(opt)
	model_path = os.path.join('checkpoints', opt.name, opt.model + '_' + str(8) + '.pth')
	#model_path = os.path.join('checkpoints', opt.name, opt.model + '_best.pth')

	if len(opt.gpu_ids) > 1:
		model = model.module

	model.load(model_path)

	for i, inputs in enumerate(dataloader):
		print(i)
		model.set_input(inputs, mode='eval')
		ori_img = inputs['ori_img'][0]
		c, img_h, img_w = ori_img.size()
		with torch.no_grad():
			output = model.inference()
		dets = ctdet_decode(output['hm'].sigmoid_(), output['wh'], reg=output['reg'], K=100)
		dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

		detection = dets[0]
		detection = post_processing(detection, 0.5)
		detection[:, 0] = detection[:, 0] / 128 * img_w
		detection[:, 1] = detection[:, 1] / 128 * img_h
		detection[:, 2] = detection[:, 2] / 128 * img_w
		detection[:, 3] = detection[:, 3] / 128 * img_h
		
		save_path = 'results/'+'uavdtori'+'/val/'+inputs['img_name'][0]
		vis.save_preds(inputs['ori_img'], detection, save_path)		
		save_dec_path = 'results/'+'uavdtfineori'+'/valdetection/'+inputs['img_name'][0]
		vis.save_dets(detection, save_dec_path)


def fine_eval():
	torch.backends.cudnn.benchmark = True

	opt = TestOptions().parse()
	vis = Visualizer(opt)

	dataset = get_dataset(opt)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size = 1,
		shuffle=False
	)

	model = create_model(opt)
	model_path = os.path.join('checkpoints', opt.name, opt.model + '_best.pth')

	if len(opt.gpu_ids) > 1:
		model = model.module

	model.load(model_path)

	total_time = 0
	for i, inputs in enumerate(dataloader):
		print(i)
		if (isinstance(inputs['img'], list)):
			imgs = inputs['img']
			offsets = inputs['offsets']
			ori_img = inputs['ori_img']
			img_h, img_w = ori_img.size(2), ori_img.size(3)

			detections = None
			for i, img in enumerate(imgs):
				x1, y1, x2, y2 = offsets[i]

				inp = {'img': img}
				model.set_input(inp, mode='eval')
				with torch.no_grad():
					st = time.time()
					output = model.inference()
					et = time.time()
					total_time += et - st
				
				dets = ctdet_decode(output['hm'].sigmoid_(), output['wh'], reg=output['reg'], K=100)
				dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

				detection = dets[0]
				detection[:, 0] = detection[:, 0] / 128 * int(x2-x1) + int(x1)
				detection[:, 1] = detection[:, 1] / 128 * int(y2-y1) + int(y1)
				detection[:, 2] = detection[:, 2] / 128 * int(x2-x1) + int(x1)
				detection[:, 3] = detection[:, 3] / 128 * int(y2-y1) + int(y1)
				
				detections = detection if detections is None else np.concatenate([detections, detection], axis=0)
						
			save_path = 'results/'+opt.name+'/val/'+inputs['img_name'][0]
			vis.save_preds(ori_img, detections, save_path)
			save_det_path = 'results/' + opt.name+ '/valdetection/' + inputs['img_name'][0]
			vis.save_dets(detections, save_det_path)
	print('total_time: ', total_time, 'dataset length: ', len(dataset), len(dataset) / total_time)

def coarse_eval():
	torch.backends.cudnn.benchmark = True

	opt = CTestOptions().parse()
	vis = Visualizer(opt)

	dataset = get_dataset(opt)
	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size = 1,
		shuffle = False
	)

	model = create_model(opt)
	model_path = os.path.join('checkpoints', opt.name, opt.model + '_best' + '.pth')


	if len(opt.gpu_ids) > 1:
		model = model.module

	model.load(model_path)

	for i, inputs in enumerate(dataloader):
		print(i)
		model.set_input(inputs, mode='eval')
		ori_img = inputs['ori_img']
		img_h, img_w = ori_img.size(2), ori_img.size(3)

		with torch.no_grad():
			output = model.inference()

		#vis.plot_prediction(inputs['img'], output['hm'], output['wh'], output['reg'])
		dets = ctdet_decode(output['hm'].sigmoid_(), output['wh'], reg=output['reg'], K=5)
		dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

		detection = dets[0]
		detection = post_processing(detection, threshold=0.5)
		detection[:, 0] = detection[:, 0] / 128 * img_w
		detection[:, 1] = detection[:, 1] / 128 * img_h
		detection[:, 2] = detection[:, 2] / 128 * img_w
		detection[:, 3] = detection[:, 3] / 128 * img_h

		save_path = 'results/'+opt.name+'/val/'+inputs['img_name'][0]
		vis.save_preds(inputs['ori_img'], detection, save_path)
		save_dec_path = 'results/'+opt.name+'/valdetection/'+inputs['img_name'][0]
		vis.save_dets(detection, save_dec_path)

if __name__ == '__main__':
	#coarse_eval()
	fine_eval()
	#ori_eval()
	#crop_eval()
	