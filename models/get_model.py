import torch
import torch.nn as nn 

def create_model(opt):
	if opt.model == 'hourglass':
		from .Hourglass import Hourglass
		model = Hourglass()
	elif opt.model == 'dla':
		print('not implemented yet')
	elif opt.model == 'res':
		print('not implemented yet')
	else:
		raise NotImplementedError("model %s is not implemented." % (opt.model))

	model.initialize(opt)
	if (len(opt.gpu_ids) > 0):
		model = model.to(opt.gpu_ids[0])
	if (len(opt.gpu_ids) > 1):
		model = nn.DataParallel(model)
	return model