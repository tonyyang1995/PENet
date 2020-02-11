from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import os
import cv2
import math

from transform.bounding_box import BBox 
import torch
import random
import torchvision.transforms as transforms

from util.images import flip 
from util.images import get_affine_transform, affine_transform
from util.images import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from util.images import draw_dense_reg

from .BaseDataset import BaseDataset

class CoarseTrain(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		self.img_paths, self.label_paths = self._get_paths(opt.dataroot, label_name='annotations')

		self.class_name = opt.classes
		self.max_objs = 64
		self.num_classes = opt.num_classes


		self.ToTensor = transforms.ToTensor()

	def __len__(self):
		return len(self.img_paths)

	def name(self):
		return 'Coarse'

	def __getitem__(self, index):
		img_path = self.img_paths[index]
		label_path = self.label_paths[index]

		if not os.path.exists(img_path):
			assert RuntimeError("image not found")
		if not os.path.exists(label_path):
			assert RuntimeError("label not found")
		img = Image.open(img_path).convert('RGB')
		img_w, img_h = img.size
		if self.opt.dataset == 'visDrone':
			labels = np.loadtxt(self.label_paths[index], delimiter=',').reshape(-1, 8)
			bboxes = BBox.from_visDrone_coarse(labels, img.size)
		elif self.opt.dataset == 'uavDT':
			labels = np.loadtxt(self.label_paths[index], delimiter=',').reshape(-1, 6)
			bboxes = BBox.from_uavDT_coarse(labels, img.size)
		bboxes = bboxes.non_max_merge(box_size=self.opt.input_w, iou_thresh=0.2) # ROI
		
		input_h, input_w = self.opt.input_h, self.opt.input_w
		# rand crop
		if not self.opt.not_rand_crop and (img_h > input_h and img_w > input_w):
			i = np.random.randint(0, img_h - input_h)
			j = np.random.randint(0, img_w - input_w)

			img = img.crop((j, i, j + input_w, i + input_h))
			bboxes = bboxes.crop((j, i, j+input_w, i+input_h))
		else:
			img = img.resize((input_h, input_w))

		#bboxes = bboxes.non_max_merge(box_size=(self.opt.input_w / 4, self.opt.input_h / 4), iou_thresh=0.2)

		flipped = False
		if np.random.random() < self.opt.flip:
			flipped = True
			img = transforms.functional.hflip(img)
			bboxes = bboxes.hflip()

		img = self.ToTensor(img)
		targets = bboxes.to_tensor()

		num_objs = min(self.max_objs, len(targets))
		output_h, output_w = input_h // 4, input_w // 4

		hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
		wh = np.zeros((self.max_objs, 2), dtype=np.float32)
		reg = np.zeros((self.max_objs, 2), dtype=np.float32)
		ind = np.zeros((self.max_objs), dtype=np.int64)
		reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

		for k in range(num_objs):
			#print(targets[k, ...])
			cls_id, cx, cy, w, h = targets[k]
			cls_id = int(cls_id)

			cx = (cx * output_w).numpy()
			cy = (cy * output_h).numpy()
			w = (w * output_w).numpy()
			h = (h * output_h).numpy()

			radius = gaussian_radius((math.ceil(h), math.ceil(w)))
			radius = max(0, int(radius))

			ct = np.array([cx, cy], dtype=np.float32)
			ct_int = ct.astype(np.int32)
			draw_umich_gaussian(hm[cls_id], ct_int, radius)
			wh[k] = 1. * w, 1.* h
			ind[k] = ct_int[1] * output_w + ct_int[0]
			reg[k] = ct - ct_int
			reg_mask[k] = 1

		ret = {'img': img, 'hm': hm, 'wh': wh, 'reg': reg, 'ind': ind, 'reg_mask': reg_mask}
		#ret = {'img': img, 'hm': hm, 'wh': wh, 'reg': reg, 'ind': ind, 'reg_mask': reg_mask, 'targets': targets}
		return ret

# class CoarseTrain(BaseDataset):
# 	def __init__(self, opt):
# 		self.opt = opt
# 		self.img_paths, self.label_paths = self._get_paths(opt.dataroot, label_name='gen_annotations')

# 		self.class_name = opt.classes
# 		self.max_objs = 4
# 		self.num_classes = opt.num_classes
# 		self.ToTensor = transforms.ToTensor()

# 	def __len__(self):
# 		return len(self.img_paths)

# 	def name(self):
# 		return 'Coarse'

# 	def __getitem__(self, index):
# 		img_path = self.img_paths[index]
# 		label_path = self.label_paths[index]

# 		if not os.path.exists(img_path):
# 			assert RuntimeError("image not found")
# 		if not os.path.exists(label_path):
# 			assert RuntimeError("label not found")

# 		img = Image.open(img_path).convert('RGB')
# 		img_w, img_h = img.size
# 		labels = np.loadtxt(self.label_paths[index], delimiter=',').reshape(-1, 8)
# 		bboxes = BBox.from_visDrone(labels, img.size)

# 		input_h, input_w = self.opt.input_h, self.opt.input_w
# 		# rand crop
# 		if not self.opt.not_rand_crop and (img_h > input_h and img_w > input_w):
# 			#i, j, h, w = transforms.RandomCrop.get_params(img, (input_h, input_w))
# 			# check whether the parameters are out of bound
# 			i = np.random.randint(0, img_h - input_h)
# 			j = np.random.randint(0, img_w - input_w)
# 			#img = transforms.functional.crop(img, i, j, input_h, input_w)
# 			img = img.crop((j, i, j + input_w, i + input_h))
# 			bboxes = bboxes.crop((j, i, j+input_w, i+input_h))
# 		else:
# 			img = img.resize((input_h, input_w))

# 		flipped = False
# 		if np.random.random() < self.opt.flip:
# 			flipped = True
# 			img = transforms.functional.hflip(img)
# 			bboxes = bboxes.hflip()

# 		img = self.ToTensor(img)
# 		targets = bboxes.to_tensor()

# 		num_objs = min(self.max_objs, len(targets))
# 		output_h, output_w = input_h // 4, input_w // 4

# 		hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
# 		wh = np.zeros((self.max_objs, 2), dtype=np.float32)
# 		reg = np.zeros((self.max_objs, 2), dtype=np.float32)
# 		ind = np.zeros((self.max_objs), dtype=np.int64)
# 		reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

# 		for k in range(num_objs):
# 			#print(targets[k, ...])
# 			cls_id, cx, cy, w, h = targets[k]
# 			cls_id = int(cls_id)
# 			cx = (cx * output_w).numpy()
# 			cy = (cy * output_h).numpy()
# 			w = (w * output_w).numpy()
# 			h = (h * output_h).numpy()

# 			radius = gaussian_radius((math.ceil(h), math.ceil(w)))
# 			radius = max(0, int(radius))

# 			ct = np.array([cx, cy], dtype=np.float32)
# 			ct_int = ct.astype(np.int32)
# 			draw_umich_gaussian(hm[cls_id], ct_int, radius)
# 			wh[k] = 1. * w, 1.* h
# 			ind[k] = ct_int[1] * output_w + ct_int[0]
# 			reg[k] = ct - ct_int
# 			reg_mask[k] = 1

# 		ret = {'img': img, 'hm': hm, 'wh': wh, 'reg': reg, 'ind': ind, 'reg_mask': reg_mask}
# 		return ret