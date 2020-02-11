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

class crop(BaseDataset):
	def __init__(self, opt):
		self.opt = opt
		self.img_paths = self._get_paths(opt.dataroot, label_name=None)
		self.num_classes = opt.classes 
		self.ToTensor = transforms.ToTensor()

	def name(self):
		return 'crop'

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		imgpath = self.img_paths[index]
		imgname = imgpath.split('/')[-1]
		img = Image.open(imgpath).convert('RGB')
		ret = {"ori_img": self.ToTensor(img)}

		img_w, img_h = img.size
		imgs = []
		offset = []
		for i in range(img_w // 512 + 1):
			for j in range(img_h // 512 + 1):
				left = i * 512
				top = j * 512
				right = min(left + 512, img_w)
				bot = min(top + 512, img_h)

				if right == img_w:
					left = right - 512
				if bot == img_h:
					top = bot - 512

				crop_img = img.crop((left, top, right, bot))
				imgs.append(self.ToTensor(crop_img))
				offset.append([left, top])

		#return {'imgs': imgs, 'img_name': imgname, 'offset': offset}
		ret['imgs'] = imgs
		ret['img_name'] = imgname
		ret['offset'] = offset
		return ret
