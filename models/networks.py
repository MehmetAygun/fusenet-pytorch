import torch
import torch.nn as nn
from torch.nn import init
import torchvision 
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.manual_seed(2)

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def get_scheduler(optimizer, opt):
	if opt.lr_policy == 'lambda':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', gain=0.02):
	net = net
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=gain)
			elif init_type == 'pretrained':
				pass
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, gain)
			init.constant_(m.bias.data, 0.0)
	print('initialize network with %s' % init_type)
	net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)

	for root_child in net.children():
		for children in root_child.children():
			if children in root_child.need_initialization:
				init_weights(children, init_type, gain=init_gain)
			else:
				init_weights(children,"pretrained",gain=init_gain) #for batchnorms
	return net

def define_FuseNet(rgb_enc=True, depth_enc=True, rgb_dec=True, depth_dec=False, norm='batch', use_dropout=True, init_type='xavier', init_gain=0.02, gpu_ids=[]):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	net = FusenetGenerator(rgb_enc=True, depth_enc=True, rgb_dec=True, depth_dec=False )
	return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

class FusenetGenerator(nn.Module):
	def __init__(self,rgb_enc=True, depth_enc=True, rgb_dec=True, depth_dec=False):
		super(FusenetGenerator, self).__init__()
		batchNorm_momentum = 0.1#TODO:make param
		num_labels = 40 #TODO:make parame
		
		self.need_initialization = [] #modules that need initialization
		
		if rgb_enc :
			feats_rgb = list(torchvision.models.vgg16(pretrained=True).features.children())
		
			##### RGB ENCODER ####
			self.CBR1_RGB_ENC = nn.Sequential (
			    feats_rgb[0],
			    nn.BatchNorm2d(64),
			    feats_rgb[1],
			    feats_rgb[2],
			    nn.BatchNorm2d(64),
			    feats_rgb[3],
			)

			self.CBR2_RGB_ENC = nn.Sequential (
			    feats_rgb[5],
			    nn.BatchNorm2d(128),
			    feats_rgb[6],
			    feats_rgb[7],
			    nn.BatchNorm2d(128),
			    feats_rgb[8],
			)

			self.CBR3_RGB_ENC = nn.Sequential (        
			    feats_rgb[10],
			    nn.BatchNorm2d(256),
			    feats_rgb[11],
			    feats_rgb[12],
			    nn.BatchNorm2d(256),
			    feats_rgb[13],
			    feats_rgb[14],
			    nn.BatchNorm2d(256),
			    feats_rgb[15],
			)

			self.dropout3 = nn.Dropout(p=0.5)

			self.CBR4_RGB_ENC = nn.Sequential (
			    feats_rgb[17],
			    nn.BatchNorm2d(512),
			    feats_rgb[18],
			    feats_rgb[19],
			    nn.BatchNorm2d(512),
			    feats_rgb[20],
			    feats_rgb[21],
			    nn.BatchNorm2d(512),
			    feats_rgb[22],
			)

			self.dropout4 = nn.Dropout(p=0.5)

			self.CBR5_RGB_ENC = nn.Sequential (        
			    feats_rgb[24],
			    nn.BatchNorm2d(512),
			    feats_rgb[25],
			    feats_rgb[26],
			    nn.BatchNorm2d(512),
			    feats_rgb[27],
			    feats_rgb[28],
			    nn.BatchNorm2d(512),
			    feats_rgb[29],
			)

			self.dropout5 = nn.Dropout(p=0.5)
		

		if depth_enc :

			feats_depth = list(torchvision.models.vgg16(pretrained=True).features.children())
			avg = torch.mean(feats_depth[0].weight.data, dim=1)
			avg = avg.unsqueeze(1)
			
			self.conv11d = nn.Conv2d(1, 64, kernel_size=3,padding=1)
			self.conv11d.weight.data = avg 

			self.CBR1_DEPTH_ENC = nn.Sequential(
			    nn.BatchNorm2d(64),
			    feats_depth[1],
			    feats_depth[2],
			    nn.BatchNorm2d(64),
			    feats_depth[3],
			)
			self.CBR2_DEPTH_ENC = nn.Sequential(
			    feats_depth[5],
			    nn.BatchNorm2d(128),
			    feats_depth[6],
			    feats_depth[7],
			    nn.BatchNorm2d(128),
			    feats_depth[8],
			)
			self.CBR3_DEPTH_ENC = nn.Sequential(
			    feats_depth[10],
			    nn.BatchNorm2d(256),
			    feats_depth[11],
			    feats_depth[12],
			    nn.BatchNorm2d(256),
			    feats_depth[13],
			    feats_depth[14],
			    nn.BatchNorm2d(256),
			    feats_depth[15],
			)

			self.dropout3_d = nn.Dropout(p=0.5)

			self.CBR4_DEPTH_ENC = nn.Sequential(
			    feats_depth[17],
			    nn.BatchNorm2d(512),
			    feats_depth[18],
			    feats_depth[19],
			    nn.BatchNorm2d(512),
			    feats_depth[20],
			    feats_depth[21],
			    nn.BatchNorm2d(512),
			    feats_depth[22],
			)

			self.dropout4_d = nn.Dropout(p=0.5)

			self.CBR5_DEPTH_ENC = nn.Sequential(
			    feats_depth[24],
			    nn.BatchNorm2d(512),
			    feats_depth[25],
			    feats_depth[26],
			    nn.BatchNorm2d(512),
			    feats_depth[27],
			    feats_depth[28],
			    nn.BatchNorm2d(512),
			    feats_depth[29],
			) 
		if  rgb_dec :
		
			####  RGB DECODER  ####
			self.CBR5_RGB_DEC = nn.Sequential (        
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			)

			self.need_initialization.append(self.CBR5_RGB_DEC)

			self.CBR4_RGB_DEC = nn.Sequential (        
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Conv2d(512, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			)

			self.need_initialization.append(self.CBR4_RGB_DEC)

			self.CBR3_RGB_DEC = nn.Sequential (        
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Conv2d(256,  128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			)
					
			self.need_initialization.append(self.CBR3_RGB_DEC)

			self.CBR2_RGB_DEC = nn.Sequential (
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128, momentum= batchNorm_momentum),
			nn.ReLU(),
			nn.Conv2d(128, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64, momentum= batchNorm_momentum),
			nn.ReLU(),
			)

			self.need_initialization.append(self.CBR2_RGB_DEC)

			self.CBR1_RGB_DEC = nn.Sequential (                
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64, momentum= batchNorm_momentum),
			nn.ReLU(),        	
			nn.Conv2d(64, num_labels, kernel_size=3, padding=1),
			)
		
			self.need_initialization.append(self.CBR1_RGB_DEC)

	def forward(self, rgb_inputs,depth_inputs):
		
		########  DEPTH ENCODER  ########
		# Stage 1
		x = self.conv11d(depth_inputs)
		x_1 = self.CBR1_DEPTH_ENC(x)
		x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)
		
		# Stage 2
		x_2 = self.CBR2_DEPTH_ENC(x)
		x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)

		# Stage 3
		x_3 = self.CBR3_DEPTH_ENC(x)
		x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
		x = self.dropout3_d(x)

		# Stage 4
		x_4 = self.CBR4_DEPTH_ENC(x)
		x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
		x = self.dropout4_d(x)

		# Stage 5
		x_5 = self.CBR5_DEPTH_ENC(x)

		########  RGB ENCODER  ########

		# Stage 1
		y = self.CBR1_RGB_ENC(rgb_inputs)
		y = torch.add(y,x_1)
		y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

		# Stage 2
		y = self.CBR2_RGB_ENC(y)
		y = torch.add(y,x_2)
		y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

		# Stage 3
		y = self.CBR3_RGB_ENC(y)
		y = torch.add(y,x_3)
		y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
		y = self.dropout3(y)

		# Stage 4
		y = self.CBR4_RGB_ENC(y)
		y = torch.add(y,x_4)
		y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
		y = self.dropout4(y)

		# Stage 5
		y = self.CBR5_RGB_ENC(y)
		y = torch.add(y,x_5)
		y_size = y.size() 

		y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
		y = self.dropout5(y)

		########  DECODER  ########

		# Stage 5 dec
		y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
		y = self.CBR5_RGB_DEC(y)

		# Stage 4 dec
		y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
		y = self.CBR4_RGB_DEC(y)

		# Stage 3 dec
		y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
		y = self.CBR3_RGB_DEC(y)

		# Stage 2 dec
		y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
		y = self.CBR2_RGB_DEC(y)

		# Stage 1 dec
		y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
		y = self.CBR1_RGB_DEC(y)

		return y
		
class SegmantationLoss(nn.Module):
	def __init__(self):
		super(SegmantationLoss, self).__init__()
	        self.weights =torch.cuda.FloatTensor([0.272491, 0.568953, 0.432069, 0.354511, 0.82178, 0.506488, 1.133686, 0.81217, 0.789383, 0.380358, 1.650497, 1, 0.650831, 0.757218, 0.950049, 0.614332, 0.483815, 1.842002, 0.635787, 1.176839, 1.196984, 1.111907, 1.927519, 0.695354, 1.057833, 4.179196, 1.571971, 0.432408, 3.705966, 0.549132, 1.282043, 2.329812, 0.992398, 3.114945, 5.466101, 1.085242, 6.968411, 1.093939, 1.33652, 1.228912])	
		self.loss = nn.CrossEntropyLoss()
	def __call__(self, output, target,pixel_average=True):
		if pixel_average:
			 return self.loss(output, target) #/ target.data.sum()
		else:
			return self.loss(output, target)
		
		
