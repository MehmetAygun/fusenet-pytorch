import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class FuseNetModel(BaseModel):
	def name(self):
		return 'FuseNetModel'

	@staticmethod
	def modify_commandline_options(parser, is_train=True):

		# changing the default values 
		if is_train:
			parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
		return parser

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain
		# specify the training losses you want to print out. The program will call base_model.get_current_losses
		self.loss_names = ['segmentation']
		# specify the images you want to save/display. The program will call base_model.get_current_visuals
		self.visual_names = ['rgb_image']
		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
		
		self.model_names = ['FuseNet']
		
		# load/define networks
		self.netFuseNet = networks.define_FuseNet(rgb_enc=True, depth_enc=True, rgb_dec=True, depth_dec=False, norm=opt.norm,use_dropout= not opt.no_dropout, init_type=opt.init_type, 
												init_gain= opt.init_gain, gpu_ids= self.gpu_ids)

		if self.isTrain:
			
			# define loss functions
			self.criterionSegmentation = networks.SegmantationLoss().to(self.device)
			#self.criterionL1 = torch.nn.L1Loss()

			# initialize optimizers
			self.optimizers = []
			
			self.optimizer_FuseNet = torch.optim.Adam(self.netFuseNet.parameters(),
												lr=opt.lr, betas=(opt.beta1, 0.999))
			
			self.optimizers.append(self.optimizer_FuseNet)

	def set_input(self, input):

		self.rgb_image = input['rgb_image'].to(self.device)
		self.depth_image = input['depth_image'].to(self.device)
		self.mask = input['mask'].to(self.device)
		
	def forward(self):
		self.output = self.netFuseNet(self.rgb_image,self.depth_image)

	def backward(self):
		
		self.loss_segmentation = self.criterionSegmentation(self.output, self.mask)
	
		self.loss_segmentation.backward()

	def optimize_parameters(self):
		self.forward()
		# update Fusenet
		self.set_requires_grad(self.netFuseNet, True)
		self.optimizer_FuseNet.zero_grad()
		self.backward()
		self.optimizer_FuseNet.step()

