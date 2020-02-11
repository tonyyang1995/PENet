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

class AugFineTrain(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		self.img_paths, self.label_paths = self._get_paths(opt.dataroot, label_name='annotations')
		#_, self.gen_label_paths = self._get_paths(opt.dataroot, label_name='gen_annotations')

		self.class_name = opt.classes
		self.max_objs = 500
		self.num_classes = opt.num_classes
		self.ToTensor = transforms.ToTensor()

		if self.opt.augment:
			self.weighted = self.opt.augment_weight
			self.imgpool_path = self.opt.imgpool_path
			self.aug_paths = []
			for path in self.img_paths:
				self.aug_paths.append(path.replace('images', 'road_seg_pred'))
			self.sample_k = self.opt.sample_k
			self.file_dir = self.opt.cats

		#self.imgs, self.labels = self.get_crop_imgs()

	def __len__(self):
		#return len(self.imgs)
		return len(self.img_paths)

	def name(self):
		return 'AugFineTrain'

	def __getitem__(self, index):
		#img = self.imgs[index]
		#targets = self.labels[index]
		img_path = self.img_paths[index]
		label_path = self.label_paths[index]

		if not os.path.exists(img_path):
			assert RuntimeError("image not found")
		if not os.path.exists(label_path):
			assert RuntimeError("label not found")

		img = Image.open(img_path).convert('RGB')
		img_w, img_h = img.size
		#labels = np.loadtxt(self.label_paths[index], delimiter=',').reshape(-1, 8)
		#targets = BBox.from_visDrone(labels, img.size)
		if self.opt.dataset == 'visDrone':
			labels = np.loadtxt(self.label_paths[index], delimiter=',').reshape(-1, 8)
			targets = BBox.from_visDrone(labels, img.size)
		elif self.opt.dataset == 'uavDT':
			labels = np.loadtxt(self.label_paths[index], delimiter=',').reshape(-1, 6)
			targets = BBox.from_uavDT(labels, img.size)

		img_w, img_h = img.size
		
		rand_crop_seed = np.random.random() # set up 
		if self.opt.augment and rand_crop_seed > 0.5:
			# add random sample by weight
			for k in range(self.sample_k):
				# choose a sample
				rand = np.random.random()
				for i, p in enumerate(self.weighted):
					if rand < p:
						texture_path = os.path.join(self.imgpool_path, self.file_dir[i])
						tex_cls_id = self.opt.classes.index(self.file_dir[i])

						textures = os.listdir(texture_path)
						rand_idx = np.random.randint(0, len(textures))
						texture_path = os.path.join(texture_path, textures[rand_idx])
						texture_img = Image.open(texture_path)
						texture_w, texture_h = texture_img.size
						#texture_img = self.ToTensor(texture_img)
						# find the position
						tex_i = np.random.randint(0, img_w - texture_h)
						tex_j = np.random.randint(0, img_h - texture_w)

						img.paste(texture_img, (tex_i, tex_j, tex_i + texture_w, tex_j + texture_h))
						targets.add_gt(tex_i, tex_j, texture_h, texture_w, tex_cls_id)

		input_h, input_w = self.opt.input_h, self.opt.input_w
		rand_crop_seed = np.random.random() # set up 
		if not self.opt.not_rand_crop and (img_h > input_h and img_w > input_w) and rand_crop_seed < 0.6:
			i = np.random.randint(0, img_h - input_h)
			j = np.random.randint(0, img_w - input_w)
			img = img.crop((j, i, j + input_w, i + input_h))
			targets = targets.crop((j, i, j + input_w, i + input_h))
		else:
			img = img.resize((input_h, input_w))

		flipped = False
		if np.random.random() < self.opt.flip:
			flipped = True
			img = transforms.functional.hflip(img)
			targets = targets.hflip()

		img = self.ToTensor(img)
		targets = targets.to_tensor()

		num_objs = min(self.max_objs, len(targets))
		output_h, output_w = input_h // 4, input_w // 4

		hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
		if self.opt.stack:
			shm = np.zeros((self.opt.stack_num_classes, output_h, output_w), dtype=np.float32)
		wh = np.zeros((self.max_objs, 2), dtype=np.float32)
		reg = np.zeros((self.max_objs, 2), dtype=np.float32)
		ind = np.zeros((self.max_objs), dtype=np.int64)
		reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

		for k in range(num_objs):
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

			if self.opt.stack:
				if cls_id == 1 or cls_id == 2: # human
					draw_umich_gaussian(shm[0], ct_int, radius)
				elif cls_id == 3 or cls_id == 7 or cls_id == 8:
					draw_umich_gaussian(shm[1], ct_int, radius)
				elif cls_id == 4 or cls_id == 5 or cls_id == 6 or cls_id == 9 or cls_id == 10:
					draw_umich_gaussian(shm[2], ct_int, radius)
			wh[k] = 1. * w, 1. * h
			ind[k] = ct_int[1] * output_w + ct_int[0]
			reg[k] = ct - ct_int
			reg_mask[k] = 1
		ret = {'img': img, 'hm': hm, 'wh': wh, 'reg': reg, 'ind': ind, 'reg_mask': reg_mask}
		if self.opt.stack:
			ret['shm'] = shm
		return ret

	def get_crop_imgs(self):
		imgs = list(); labels = list()

		for i, imgpath in enumerate(self.img_paths):
			genlabelpath = self.gen_label_paths[i]
			labelpath = self.label_paths[i]

			img = Image.open(imgpath).convert('RGB')
			img_w, img_h = img.size
			genlabel = BBox.from_visDrone(np.loadtxt(genlabelpath, delimiter=',').reshape(-1, 8), img.size).to_tensor()
			label = BBox.from_visDrone(np.loadtxt(labelpath, delimiter=',').reshape(-1, 8), img.size)
	
			for gen in genlabel:
				cls_id, cx, cy, w, h = gen
				left = int(cx - w / 2 * img_w)
				top = int(cy - h / 2 * img_h)
				right = int(cx + w / 2 * img_w)
				bot = int(cy + h / 2 * img_h)

				new_img = img.crop((left, top, right, bot))
				new_label = label.crop((left, top, right, bot))

				new_target = new_label.to_tensor()
				print(new_target.size())
				if new_target.size(0) > 0:
					imgs.append(new_img)
					labels.append(new_label)

		return imgs, labels
