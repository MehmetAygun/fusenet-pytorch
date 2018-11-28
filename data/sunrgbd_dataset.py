import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import pickle

class sunrgbddataset(BaseDataset):
	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	def initialize(self, opt):
		self.opt = opt
		self.batch_size = opt.batch_size
		self.root = opt.dataroot # path for the dataset
		splits = pickle.load(open(os.path.join(self.root, "splits.pkl"), "rb"), encoding="latin1")
		self.indexes = splits["trainval"] if opt.phase == "train" else splits["test"]
		self.num_labels = 38
		self.ignore_label = 0
		self.class_weights = torch.from_numpy(np.loadtxt(os.path.join(opt.dataroot, "class_weights"),
														delimiter=',').astype(np.float32)
											)
		assert(opt.resize_or_crop == 'none')

	def __getitem__(self, index):
		index = self.indexes[index]
		rgb_image = np.array(Image.open(os.path.join(self.root, "images-224", str(index)+".png")))
		depth_image = np.array(Image.open(os.path.join(self.root, "depth-inpaint-u8-224", str(index)+".png")))
		mask = np.array(Image.open(os.path.join(self.root, "seglabel-224", str(index)+".png")))

		rgb_image = transforms.ToTensor()(rgb_image)
		depth_image = transforms.ToTensor()(depth_image[:, :, np.newaxis])

		mask = torch.from_numpy(mask)
		mask = mask.type(torch.LongTensor)

		return {'rgb_image': rgb_image, 'depth_image': depth_image, 'mask': mask, 'path': str(index)+".png"}

	def __len__(self):
		return len(self.indexes)

	def name(self):
		return 'sunrgbd'
