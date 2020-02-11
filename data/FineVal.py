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

class FineVal(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		self.img_paths, self.crop_paths = self._get_paths(opt.dataroot, label_name='valdetection')

		self.num_classes = opt.classes
		self.ToTensor = transforms.ToTensor()

	def name(self):
		return 'FineVal'

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		# here we read an image, then crop it using the middle reuslt from coarse det
		# return ori_img, crop_imgs, crop_offsets
		imgpath = self.img_paths[index]
		imgname = imgpath.split('/')[-1]

		img = Image.open(imgpath).convert('RGB')
		img_w, img_h = img.size
		ret = {'ori_img': self.ToTensor(img), 'img_name': imgname}

		crop_path = self.crop_paths[index]
		crops = np.loadtxt(crop_path).reshape(-1, 6)
		crops = crops[:, :4]

		crop_imgs = list()
		offsets = list()

		resize_img = img.resize((self.opt.input_w, self.opt.input_h))
		resize_img = self.ToTensor(resize_img)
		crop_imgs.append(resize_img)
		offsets.append([0,0,0,0])
		#print(crop_img.size())
		#ret['img'] = crop_img
		#ret['offsets'] = [0,0,0,0]

		for crop in crops:
			x1, y1, x2, y2 = crop
			# find the center point
			# crop 512  * 512
			cx = (x1 + x2) / 2
			cy = (y1 + y2) / 2
			x1 = max(0, cx - 256)
			x2 = min(cx + 256, img_w)
			y1 = max(0, cy - 256)
			y2 = min(cy + 256, img_h)
			if x1 == 0:
				x2 = 512
			if x2 == img_w:
				x1 = img_w - 512
			if y1 == 0:
				y2 = 512
			if y2 == img_h:
				y1 = img_h - 512
			x1 = int(x1)
			y1 = int(y1)
			x2 = int(x2)
			y2 = int(y2)

			# x1 = int(x1 / 512 * img_w)
			# y1 = int(y1 / 512 * img_h)
			# x2 = int(x2 / 512 * img_w)
			# y2 = int(y2 / 512 * img_h)

			crop_img = img.crop((x1, y1, x2, y2))
			crop_img = crop_img.resize((512,512))
			crop_img = self.ToTensor(crop_img)
			crop_imgs.append(crop_img)

			offset = [x1, y1, x2, y2]
			offsets.append(offset)

		ret['img'] = crop_imgs
		ret['offsets'] = offsets

		return ret