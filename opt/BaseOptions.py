import argparse
import os
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		# model and dataset
		self.parser.add_argument("--checkpoint_dir", type=str, default='checkpoints', help='checkpoint_dir')
		self.parser.add_argument('--model', type=str, default='hourglass', help='hourglass, dla, res')
		self.parser.add_argument("--dataset", default="uavDT", help="visDrone, uavDT, caltech")
		#self.parser.add_argument("--dataset", default="visDrone", help="visDrone, uavDT, caltech")

		#self.parser.add_argument("--dataset_mode", default="visDrone", help="visDrone, uavDT, caltech")
		self.parser.add_argument("--not_rand_crop", action='store_true')
		self.parser.add_argument("--shift", type=float, default=0.1)
		self.parser.add_argument("--scale", type=float, default=0.8)
		self.parser.add_argument("--rotate", type=float, default=0)
		self.parser.add_argument("--flip", type=float, default=0.5)
		self.parser.add_argument("--no_color_aug", action='store_true')
		#self.parser.add_argument("--K", type=int, default=5)

		# save info
		self.parser.add_argument("--display_freq", type=int, default=50, help='display the results in every # iterations')
		self.parser.add_argument("--val_freq", type=int, default=5, help='validation from the latest results in every iterations')

		# gpu info
		self.parser.add_argument("--gpu_ids", type=str, default='0,1')

		self.parser.add_argument("--notprint", action='store_true')
		self.initialized = True
		return self.parser

	def print_options(self, opt):
		message = ''
		message += '------------------------ OPTIONS -----------------------------\n'
		for k,v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}\n'.format(str(k), str(v), comment)
		message += '------------------------  END   ------------------------------\n'
		print(message)

		# save to the disk
		expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
		if not os.path.exists(expr_dir):
			os.makedirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write(message)
			opt_file.write('\n')

	def parse(self):
		# parse the options, create checkpoint, and set up device
		if not self.initialized:
			self.initialize()

		self.opt = self.parser.parse_args()

		if not self.opt.notprint:
			self.print_options(self.opt)
		os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
		# set gpu ids
		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)
		if len(self.opt.gpu_ids) > 0:
			self.opt.device = torch.device('cuda')
		else:
			self.opt.device = torch.device('cpu')
		return self.opt