import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
from scipy.ndimage.filters import convolve
import staintools
import torchstain
import openslide
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEP = '--'
# Tensor with channel first, and values in [0, 255]
T = transforms.Compose([
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x*255)
])

def read_slides_name(dir_source, in_child_dir=False):
	if not in_child_dir:
		slides = sorted(os.listdir(dir_source))
		slides = [slide for slide in slides if os.path.isfile(os.path.join(dir_source, slide))]
	else:
		print("Read slides from child directories.")
		slides = []

		parent_dirs = sorted(os.listdir(dir_source))
		for pdir in parent_dirs:
			path_pdir = os.path.join(dir_source, pdir)
			if not os.path.isdir(path_pdir):
				continue

			for s in os.listdir(path_pdir):
				path_slide = os.path.join(path_pdir, s)
				if not os.path.isfile(path_slide):
					continue

				slides.append("%s%s%s" % (pdir, SEP, s))

	return slides

def get_slide_id(slide, has_ext=True, in_child_dir=False):
	if has_ext:
		slide_id, _ = os.path.splitext(slide)
	else:
		slide_id = slide

	if in_child_dir:
		slide_id = slide_id.split(SEP)[-1]

	return slide_id

def get_slide_fullpath(dir_source, slide, in_child_dir=False):
	if not in_child_dir:
		full_path = os.path.join(dir_source, slide)
	else:
		pdir, slide_name = slide.split(SEP)
		full_path = os.path.join(dir_source, pdir, slide_name)

	return full_path

def img_energy(img):
	filter_du = np.array([
		[1.0, 2.0, 1.0],
		[0.0, 0.0, 0.0],
		[-1.0, -2.0, -1.0],
	])
	filter_du = np.stack([filter_du] * 3, axis=2)

	filter_dv = np.array([
		[1.0, 0.0, -1.0],
		[2.0, 0.0, -2.0],
		[1.0, 0.0, -1.0],
	])
	filter_dv = np.stack([filter_dv] * 3, axis=2)

	img = img.astype('float32')
	convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
	# energy_maps = convolved.sum(axis=2)
	energy = convolved.sum()
	return energy

def calc_patches_energy(wsi_object, patch_coords, patch_level, patch_size):
	energy = []
	for coord in patch_coords:
		try:
			img = np.array(wsi_object.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB'))
			eg = img_energy(img)
		except openslide.lowlevel.OpenSlideError as e:
			eg = 0.0
		energy.append(eg)

	return energy

def color_normalization(img, normalizer):
	"""
	:img: numpy.ndarray with shape of [H, W, C]
	"normalizer: class torchstain.MacenkoNormalizer

	:return
	    torch.Tensor() with shape of [C, H, W]
	"""
	img = staintools.LuminosityStandardizer.standardize(img)
	try:
		rimg = normalizer.normalize(T(img), stains=False)[0]
	except Exception as e:
		print("skiped color norm.")
		rimg = torch.from_numpy(img)
	# put channel first: [C, H, W]
	rimg = rimg.transpose(0, 2).transpose(1, 2)

	return rimg

def get_color_normalizer(size):
	template_path = './docs/template-%d.jpg' % size
	print("[INFO] Color Normalizer: template image is from %s" % template_path)
	target = staintools.read_image(template_path)
	target = staintools.LuminosityStandardizer.standardize(target)
	torch_normalizer = torchstain.MacenkoNormalizer(backend='torch')
	torch_normalizer.fit(T(target))

	return torch_normalizer

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_MIL, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate_MIL, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate_MIL, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

