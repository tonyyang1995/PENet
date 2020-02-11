import os
import torch
from .BaseOptions import BaseOptions

class TestOptions(BaseOptions):
	def initialize(self):
		self.parser = BaseOptions.initialize(self)

		self.parser.add_argument("--num_classes", default=2, type=int)
		self.parser.add_argument("--name", type=str, default='uavdt_coarse')

		self.parser.add_argument("--dataroot", type=str, default='dataset/UAVDT/val')
		self.parser.add_argument("--dataset_mode", default="CoarseVal", help="visDrone, uavDT, caltech")

		# batch info
		#self.parser.add_argument("--start_epochs", type=int, default=0, help="start from which epochs")
		#self.parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
		self.parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")

		# load model
		self.parser.add_argument("--load_model", default='')
		self.parser.add_argument("--resume", action='store_true')
		self.parser.add_argument("--iter_name", type=str, default='latest.pth')
		self.parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")

		# lr
		self.parser.add_argument("--lr", type=float, default=1.25e-4)
		self.parser.add_argument("--lr_step", type=str, default='90,120')

		# input
		#self.parser.add_argument("--input_res", type=int, default=512)
		self.parser.add_argument("--input_h", type=int, default=512)
		self.parser.add_argument("--input_w", type=int, default=512)

		# loss
		self.parser.add_argument('--reg_loss', default='l1', help='sl1, l1, l2')
		self.parser.add_argument('--hm_weight', type=float, default=1, help='loss weight for keypoint heatmap')
		self.parser.add_argument('--off_weight', type=float, default=1, help='loss weight for keypoint local offsets')
		self.parser.add_argument('--wh_weight', type=float, default=0.1, help='loss weight for bounding box size')

		# norms
		self.parser.add_argument('--norm_wh', action='store_true')
		self.parser.add_argument('--dense_wh', action='store_true')
		self.parser.add_argument('--cat_spec_wh', action='store_true')
		self.parser.add_argument('--not_reg_offset', action='store_true')

		# model channels info
		self.parser.add_argument('--head_conv', type=int, default=64, help='64 for resnets, 256 for dla.')
		self.parser.add_argument('--down_ratio', type=int, default=4, help='output stride, only support 4 for now')

		self.parser.add_argument('--stack', action='store_true')


		return self.parser

	def parse(self):
		# parse the options, create checkpoint, and set up device
		if not self.initialized:
			self.initialize()

		self.opt = self.parser.parse_args()

		#self.print_options(self.opt)
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

		self.opt.pad = 127 if 'hourglass' in self.opt.model else 31
		self.opt.lr_step = [int(i) for i in self.opt.lr_step.split(',')]
		self.opt.reg_offset = not self.opt.not_reg_offset
		#opt.reg_bbox = not opt.reg_bbox
		#opt.hm_hp = not opt.not_hm_hp
		#opt.reg_hp_offset = (not opt.not_reg_hp_offset) and opt.hm_hp

		self.opt.num_stacks = 2 if 'hourglass' in self.opt.model else 1

		self.opt.heads = {'hm': self.opt.num_classes, 'wh': 2, 'reg': 2}

		self.opt.classes = ['background', 'cluster']

		self.opt.not_rand_crop = True

		return self.opt
