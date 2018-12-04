import os.path
import glob
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import scipy.io as sio
import cv2

class Scannetv2Dataset(BaseDataset):
	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	def initialize(self, opt):
		self.opt = opt
		self.batch_size = opt.batch_size
		self.root = opt.dataroot
		self.num_labels = 41
		self.ignore_label = 0
		self.class_weights = None
		with open('./datasets/scannet/scannetv2_weigths.txt') as f:
			weights = f.readlines()
		self.class_weights = torch.from_numpy(np.array([float(x.strip()) for x in weights]))
		self.class_weights = self.class_weights.type(torch.FloatTensor)

		with open('./datasets/scannet/scannetv2_{}.txt'.format(opt.phase)) as f:
			scans = f.readlines()
		self.scans = [x.strip() for x in scans]

		self.rgb_frames = []
		self.depth_frames = []
		self.masks = []

		self.total_frames = 0
		for scan in self.scans:
			rgb_frames = glob.glob("{}/{}/color/*.jpg".format(self.root, scan))
			depth_frames = glob.glob("{}/{}/depth/*.png".format(self.root, scan))
			masks = glob.glob("{}/{}/label/*.png".format(self.root, scan))
			if len(rgb_frames) == len(depth_frames):
				rgb_frames.sort()
				depth_frames.sort()
				masks.sort()
				self.total_frames += len(rgb_frames)
				self.rgb_frames.extend(rgb_frames)
				self.depth_frames.extend(depth_frames)
				self.masks.extend(masks)

	def __getitem__(self, index):

		size = (320,240)
		rgb_image = np.array(Image.open(self.rgb_frames[index]))
		rgb_image = cv2.resize(rgb_image, size, interpolation=cv2.INTER_LINEAR)
		depth_image = np.array(Image.open(self.depth_frames[index]))
		depth_image = cv2.resize(depth_image, size,interpolation=cv2.INTER_NEAREST).astype(np.float)
		depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 255
		depth_image = depth_image.astype(np.uint8)
		mask_fullsize = []
		if self.masks:
			mask = np.array(Image.open(self.masks[index]))
			mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
			if self.opt.phase == "val":
				mask_fullsize = np.array(Image.open(self.masks[index]))
		else: # test phase
			mask = np.zeros((240, 320), dtype=int)
			mask_fullsize = np.zeros((968, 1296), dtype=int)

		rgb_image = transforms.ToTensor()(rgb_image)
		rgb_image = rgb_image.type(torch.FloatTensor)
		depth_image = transforms.ToTensor()(depth_image[:, :, np.newaxis])
		depth_image = depth_image.type(torch.FloatTensor)

		mask = torch.from_numpy(mask)
		mask = mask.type(torch.LongTensor)

		return {'rgb_image': rgb_image, 'depth_image': depth_image,
		        'mask': mask, 'mask_fullsize': mask_fullsize,
				'path': self.depth_frames[index].split('/')[-1], 'scan':self.depth_frames[index].split('/')[-3]}

	def __len__(self):
		return self.total_frames

	def name(self):
		return 'Scannetv2'
