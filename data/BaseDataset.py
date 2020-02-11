import random 
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
import os
import cv2

class BaseDataset(data.Dataset):
	def __init__(self, opt):
		super(BaseDataset, self).__init__()

	def name(self):
		return "BaseDataset"

	def _convert_bbox(self, x,y,w,h):
		bbox = np.array(x, y, x + w, y + h, dtype=np.float32)
		return bbox

	def _get_border(self, border, size):
		i = 1
		while size - border // i <= border // i:
			i *= 2
		return border // i 

	def _get_paths(self, dataroot, suffix='.jpg', label_name='annotations'):
		img_paths = []
		label_paths = []
		for path, subpath, files in os.walk(dataroot):
			for file in files:
				if suffix in file:
					img_path = '/'.join([path, file])
					if label_name is not None:
						label_path = img_path.replace('images', label_name)
						label_path = label_path.replace('jpg', 'txt')

					img_paths.append(img_path)
					
					if label_name is not None:
						label_paths.append(label_path)
		if label_name is not None:
			return img_paths, label_paths
		else:
			return img_paths

	def pad_to_square(self, img, pad_value):
		# this function makes an image into square shape
		c, h, w = img.shape
		dim_diff = np.abs(h-w)
		pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
		pad = (0,0, pad1, pad2) if h <= w else (pad1, pad2, 0,0)
		img = F.pad(img, pad, 'constant', value=pad_value)
		return img, pad

	def resize(self, image, size, mode='bilinear'):
		img_size = image.size()
		#print("img size is: ", len(img_size))
		if (len(img_size) == 3):
			image = image.unsqueeze(0)
		image = F.interpolate(image, size=size, mode=mode).squeeze(0)
		return image 