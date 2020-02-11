import numpy as np
import os
import visdom

from models.decode import ctdet_decode, ctdet_decode2
from .images import transform_preds

import torch
import torchvision.transforms as transforms

from PIL import Image, ImageFont
from PIL.ImageDraw import Draw 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import cv2
import random

from util.evaluate import nms

def clean_matplot():
	fig, ax = plt.subplots(1)
	plt.axis('off')
	plt.gca().xaxis.set_major_locator(NullLocator())
	plt.gca().yaxis.set_major_locator(NullLocator())
	plt.tight_layout(pad=0)
	return fig, ax

class Visualizer():
	def __init__(self, opt, theme='black', env='main'):
		self.opt = opt
		# init a visdom server
		self.viz = visdom.Visdom(port=8097, env=env)

		self.color_map = plt.get_cmap('tab20b')
		self.colors = [self.color_map(i) for i in np.linspace(0, 1, 20)]

		plt.figure()

		self.log_root = os.path.join(opt.checkpoint_dir, opt.name)
		if not os.path.exists(self.log_root):
			os.makedirs(self.log_root)

		self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
		with open(self.log_name, 'a') as log_file:
			log_file.write('==================================================== Train Loss ===============================\n')

		self.plot = False
		self.class_name = opt.classes
	
	def _del(self):
		plt.close()

	def plot_crop_images(self, imgs):
		self.viz.images(imgs)

	def plot_rawdata_target(self, imgs, targets, env='main'):
		# only plot the index 0 for debug
		img = imgs[0]
		target = targets[0]
		c, height, width = img.size()
		# draw the box img
		img = img.numpy().transpose((1,2,0))
		fig, ax = clean_matplot()
		ax.imshow(img)
		for cls_id, cx, cy, w, h in target:
			color = self.colors[int(cls_id)]
			cls_id = int(cls_id)
			if cls_id == 0 or cls_id == 11:
				cls_id = 0
			else:
				cls_id = 1
			x1 = (cx - w / 2) * width
			y1 = (cy - h / 2) * height
			box_w = w * width
			box_h = h * height
			bbox = patches.Rectangle((x1,y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
			ax.add_patch(bbox)
			plt.text(
				x1, y1, self.class_name[int(cls_id)],
				color = 'white',
				verticalalignment = 'top',
				bbox = {'color': color, 'pad': 0}
			)

		name = 'gt_index0'

		self.viz.matplot(plt, win='gt', env=env, opts=dict(title=name))

	def plot_heatmap(self, imgs, hms, env='main', name='gt_hm'):
		img = imgs[0].numpy().transpose((1,2,0)) * 255
		hm = hms[0].clone()[:12,...]
		hm = hm.detach()
		hm = hm.numpy().transpose((1,2,0)) * 255
		c = hm.shape[2]
		color_array = []
		for i in range(c):
			color = list(self.colors[i])
			color = color[:-1]
			#color = [1. - co for co in color]
			color_array.append(color)

		color_array = np.asarray(color_array).reshape(1,1,c,3)
		hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2], 1)
		hm_color_map = (hm * color_array).max(axis=2).astype(np.uint8)

		if (img.shape[0] != hm.shape[0]):
			hm_color_map = cv2.resize(hm_color_map, (img.shape[0], img.shape[1]))
		blend_img = (255 - hm_color_map) * 1 + img * 0
		blend_img[blend_img > 255] = 255
		blend_img[blend_img < 0] = 0

		blend_img = blend_img.astype(np.uint8).copy()
		
		self.viz.image(blend_img.transpose((2,0,1)), win=name, opts=dict(caption=name))
		plt.close('all')

	def print_loss(self, error_ret, epoch, cur_iter, total_iter):
		if self.opt.stack:
			metrics = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'shm_loss']
		else:
			metrics = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
		message = '\n----------[Epoch %d/%d, Batch %d/%d] -----------------\n' % (epoch, self.opt.start_epochs+self.opt.epochs, cur_iter, total_iter)
		
		for key in metrics:
			message += '{:>10}\t{:>10.4f}\n'.format(key, error_ret[key])
		message += '------------------------------------------------------\n'
		print(message)
		with open(self.log_name, 'a') as log_file:
			log_file.write('%s\n' % message)
	

	# plot prediction
	def plot_prediction(self, img, hm, wh, reg):
		dets = ctdet_decode(hm.sigmoid_(), wh, reg=reg, K=100)
		dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
		img = img[0]
		detection = dets[0]
		detection = nms(detection)
		img = img.numpy().transpose((1,2,0))
		fig, ax = clean_matplot()
		ax.imshow(img)

		for bboxes in detection:
			x1, y1, x2, y2 = bboxes[:4] * 4# xyxy
			cls_id = bboxes[-1]
			score = bboxes[4]

			if score < 0.5:
				continue
			if cls_id == 0 or cls_id > 11:
				continue
			color = self.colors[int(cls_id)]
			box_w = x2 - x1
			box_h = y2 - y1
			if box_w < 0.1 or box_h < 0.1:
				continue
			
			bbox = patches.Rectangle((x1,y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
			ax.add_patch(bbox)
			plt.text(
				x1, y1, self.class_name[int(cls_id)]+ '=%.1f' % score,
				color = 'white',
				verticalalignment = 'top',
				bbox = {'color': color, 'pad': 0}
			)

		name = 'pred_index0'
		self.viz.matplot(plt, win='pred', opts=dict(title=name))
		plt.close()

	def save2path(self, img, hm, wh, reg, paths):
		dets = ctdet_decode2(hm.sigmoid_(), reg=reg, K=100)
		dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

		img = img[0]
		detection = dets[0]
		detection = nms(detection)

		img = img.numpy().transpose((1,2,0))
		fig, ax = clean_matplot()

		for bboxes in detection:
			x1, y1 ,x2, y2 = bboxes[:4] * 4
			cls_id = bboxes[-1]
			score = bboxes[4]

			if cls_id == 0 or cls_id == 11:
				continue
			color = self.colors[int(cls_id)]
			box_w = x2 - x1
			box_h = y2 - y1

			bbox = patches.Rectangle((x1,y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
			ax.add_patch(bbox)
			plt.text(
				x1,y1, self.class_name[int(cls_id)] + '=%.1f' % score,
				color = 'white',
				verticalalignment = 'top',
				bbox = {'color': color, 'pad': 0}
			)

		plt.savefig(paths)

	def save_preds(self, img, detections, paths):
		#print(img.size())
		img = img[0]
		img = img.numpy().transpose((1,2,0))

		fig, ax = clean_matplot()
		ax.imshow(img)

		for bboxes in detections:
			x1, y1, x2, y2 = bboxes[:4]
			cls_id = bboxes[-1]
			score = bboxes[4]

			if score - 0 < 0.5:
				continue
			if cls_id == 0 or cls_id == 11:
				continue

			color = self.colors[int(cls_id)]
			box_w = int(x2 - x1)
			box_h = int(y2 - y1)
			if box_w < 0.1 or box_h < 0.1:
				continue

			bbox = patches.Rectangle((int(x1),int(y1)), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
			ax.add_patch(bbox)
			plt.text(
				x1,y1, self.class_name[int(cls_id)] + '=%.1f' % score,
				color = 'white',
				verticalalignment = 'top',
				bbox = {'color': color, 'pad': 0}
			)

		plt.savefig(paths)
		plt.close()

	def save_dets(self, detections, paths):
		paths = paths.replace('jpg', 'txt')
		file = open(paths, 'w')

		for bboxes in detections:
			x1, y1, x2, y2 = bboxes[:4]
			x1 = max(float(x1), 0.0)
			y1 = max(float(y1), 0.0)
			x2 = float(x2)
			y2 = float(y2)
			box_w = float(x2 - x1)
			box_h = float(y2 - y1)

			cls_id = int(bboxes[-1])
			score = bboxes[4]

			if score - 0 < 0.5:
				continue
			if cls_id == 0 or cls_id == 11:
				continue

			if box_w < 0.1 or box_h < 0.1:
				continue

			lines = str(x1) +' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(cls_id) + ' ' + str(score) + '\n'
			file.write(lines)

	def plot_loss(self, loss_stat, epochs, iter, total_iter):
		x = epochs + iter/ total_iter
		loss = loss_stat['loss']
		hm_loss = loss_stat['hm_loss']
		wh_loss = loss_stat['wh_loss']
		reg_loss = loss_stat['off_loss']
		if self.opt.stack:
			shm_loss = loss_stat['shm_loss']
		if self.plot == False:
			if self.opt.stack:
				self.viz.line(X=np.column_stack((x,x,x,x,x)), 
				Y = np.column_stack((float(loss), float(hm_loss), float(wh_loss), float(reg_loss), float(shm_loss))), opts=dict(
				title='losses', xlabels='Batchs', ylabel='losses', legend=['loss', 'hm_loss', 'wh_loss', 'off_loss', 'shm_loss']), 
				win='loss')
			else:
				self.viz.line(X=np.column_stack((x,x,x,x)), 
				Y = np.column_stack((float(loss), float(hm_loss), float(wh_loss), float(reg_loss))), opts=dict(
				title='losses', xlabels='Batchs', ylabel='losses', legend=['loss', 'hm_loss', 'wh_loss', 'off_loss']), 
				win='loss')
			self.plot = True
		else:
			if self.opt.stack:
				self.viz.line(X=np.column_stack((x,x,x,x,x)), 
				Y = np.column_stack((float(loss), float(hm_loss), float(wh_loss), float(reg_loss), float(shm_loss))), opts=dict(
				title='losses', xlabels='Batchs', ylabel='losses', legend=['loss', 'hm_loss', 'wh_loss', 'off_loss', 'shm_loss']), 
				win='loss', update='append')
			else:
				self.viz.line(X=np.column_stack((x,x,x,x)), 
				Y = np.column_stack((float(loss), float(hm_loss), float(wh_loss), float(reg_loss))), opts=dict(
				title='losses', xlabels='Batchs', ylabel='losses', legend=['loss', 'hm_loss', 'wh_loss', 'off_loss']), 
				win='loss', update='append')
