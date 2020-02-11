def get_dataset(opt):
	dataset = None
	name = opt.dataset_mode
	print(name)
	if name == 'Coarse':
		from data.CoarseTrain import CoarseTrain
		dataset = CoarseTrain(opt)
	elif name == 'Fine':
		from data.FineTrain import FineTrain
		dataset = FineTrain(opt)
	elif name == 'CoarseVal':
		from data.CoarseVal import CoarseVal
		dataset = CoarseVal(opt)
	elif name == 'FineVal':
		from data.FineVal import FineVal
		dataset = FineVal(opt)
	elif name == 'CenterVal':
		from data.CenterVal import CenterVal
		dataset = CenterVal(opt)
	elif name == 'crop':
		from data.crop import crop
		dataset = crop(opt)
	elif name == 'aug':
		from data.AugFineTrain import AugFineTrain
	else:
		raise NotImplementedError('the dataset [%s] is not implemented' % name)

	print('dataset [%s] was created' % (dataset.name()))
	return dataset