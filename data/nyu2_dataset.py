import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class Nyu2Dataset(BaseDataset):
	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot # path for nyu2.npy
		self.nyu2 = np.load("{}/{}".format(self.root,"nyu2.npy")).tolist()
		assert(opt.resize_or_crop == 'resize_and_crop')

	def __getitem__(self, index):
		
		rgb_image = self.nyu2["rgb_images"][index]
		depth_image = self.nyu2["depth_images"][index]
		mask = self.nyu2["masks"][index]
						
		rgb_image = transforms.ToTensor()(rgb_image)
		depth_image = transforms.ToTensor()(depth_image)
		mask = transforms.ToTensor()(mask)
		
		#Random flip ? 
		#if (not self.opt.no_flip) and random.random() < 0.5:
		#	idx = [i for i in range(A.size(2) - 1, -1, -1)]
		#	idx = torch.LongTensor(idx)
		#	A = A.index_select(2, idx)
		#	B = B.index_select(2, idx)

		return {'rgb_image': rgb_image, 'depth_image': depth_image, 'mask': mask}

	def __len__(self):
		return len(self.nyu2["masks"])

	def name(self):
		return 'Nyu2Dataset'