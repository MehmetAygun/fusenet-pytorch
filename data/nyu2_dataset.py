import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import scipy.io as sio

class Nyu2Dataset(BaseDataset):
	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	def initialize(self, opt):
		self.opt = opt
		self.batch_size = opt.batch_size
		self.root = opt.dataroot # path for nyu2.npy
		self.nyu2 = np.load("{}/{}".format(self.root,"nyu2.npy"),encoding = 'latin1').tolist()
		splits = sio.loadmat("{}/{}".format(self.root,"splits.mat"))
		self.indexes = [x[0] - 1  for x in splits["trainNdxs"]] if opt.phase == "train" else [x[0] -1 for x in splits["testNdxs"]]
		self.num_labels = 40
		self.ignore_label = 0
		self.class_weights = None

	def __getitem__(self, index):
		index = self.indexes[index]
		rgb_image = np.array(self.nyu2["rgb_images"][index],dtype=np.uint8)
		depth_image = self.nyu2["depth_images"][index]
		depth_image = np.expand_dims(depth_image,axis=2)
		mask = np.array(self.nyu2["masks"][index],dtype=np.uint8)

		rgb_image = transforms.ToTensor()(rgb_image)
		depth_image = transforms.ToTensor()(depth_image)

		mask = torch.from_numpy(mask)
		mask = mask.type(torch.LongTensor)

		#Random flip ?
		#if (not self.opt.no_flip) and random.random() < 0.5:
		#	idx = [i for i in range(A.size(2) - 1, -1, -1)]
		#	idx = torch.LongTensor(idx)
		#	A = A.index_select(2, idx)
		#	B = B.index_select(2, idx)
		#mask = mask.unsqueeze(0)

		return {'rgb_image': rgb_image, 'depth_image': depth_image, 'mask': mask}

	def __len__(self):
		return len(self.indexes)

	def name(self):
		return 'Nyu2Dataset'
