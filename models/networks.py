import torch
import torch.nn as nn
from torch.nn import init
import torchvision 
import functools
from torch.optim import lr_scheduler

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
	init_weights(net, init_type, gain=init_gain)
	return net

def define_FuseNet(rgb_e=True, depth_e=True, rgb_d=True, depth_d=False, norm='batch', use_dropout=True, init_type='pretrained', init_gain=0.02, gpu_ids=[]):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	net = FusenetGenerator(rgb_e=True, depth_e=True, rgb_d=True, depth_d=False )
	return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

class FusenetGenerator(nn.Module):
	def __init__(self,rgb_e=True, depth_e=True, rgb_d=True, depth_d=False,):
		super(FusenetGenerator, self).__init__()
		if rgb_e :
			feats_rgb = list(models.vgg16(pretrained=True).features.children())
		
			##### RGB ENCODER ####
			self.CBR1_RGB = nn.Sequential (
			    feats_rgb[0].cuda(gpu_device),
			    nn.BatchNorm2d(64).cuda(gpu_device),
			    feats_rgb[1].cuda(gpu_device),
			    feats_rgb[2].cuda(gpu_device),
			    nn.BatchNorm2d(64).cuda(gpu_device),
			    feats_rgb[3].cuda(gpu_device),
			)

			self.CBR2_RGB = nn.Sequential (
			    feats_rgb[5].cuda(gpu_device),
			    nn.BatchNorm2d(128).cuda(gpu_device),
			    feats_rgb[6].cuda(gpu_device),
			    feats_rgb[7].cuda(gpu_device),
			    nn.BatchNorm2d(128).cuda(gpu_device),
			    feats_rgb[8].cuda(gpu_device),
			)

			self.CBR3_RGB = nn.Sequential (        
			    feats_rgb[10].cuda(gpu_device),
			    nn.BatchNorm2d(256).cuda(gpu_device),
			    feats_rgb[11].cuda(gpu_device),
			    feats_rgb[12].cuda(gpu_device),
			    nn.BatchNorm2d(256).cuda(gpu_device),
			    feats_rgb[13].cuda(gpu_device),
			    feats_rgb[14].cuda(gpu_device),
			    nn.BatchNorm2d(256).cuda(gpu_device),
			    feats_rgb[15].cuda(gpu_device),
			)

			self.dropout3 = nn.Dropout(p=0.5).cuda(gpu_device)

			self.CBR4_RGB = nn.Sequential (
			    feats_rgb[17].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats_rgb[18].cuda(gpu_device),
			    feats_rgb[19].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats_rgb[20].cuda(gpu_device),
			    feats_rgb[21].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats_rgb[22].cuda(gpu_device),
			)

			self.dropout4 = nn.Dropout(p=0.5).cuda(gpu_device)

			self.CBR5_RGB = nn.Sequential (        
			    feats_rgb[24].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats_rgb[25].cuda(gpu_device),
			    feats_rgb[26].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats_rgb[27].cuda(gpu_device),
			    feats_rgb[28].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats_rgb[29].cuda(gpu_device),
			)

			self.dropout5 = nn.Dropout(p=0.5).cuda(gpu_device)
		

		if depth_e :

			feats_depth = list(models.vgg16(pretrained=True).features.children())
			avg = torch.mean(feats_depth[0].cuda(gpu_device).weight.data, dim=1)
		
			self.conv11d = nn.Conv2d(1, 64, kernel_size=3, padding=1).cuda(gpu_device)
			self.conv11d.weight.data = avg 

			self.CBR1_D = nn.Sequential(
			    nn.BatchNorm2d(64).cuda(gpu_device),
			    feats[1].cuda(gpu_device),
			    feats[2].cuda(gpu_device),
			    nn.BatchNorm2d(64).cuda(gpu_device),
			    feats[3].cuda(gpu_device),
			)
			self.CBR2_D = nn.Sequential(
			    feats[5].cuda(gpu_device),
			    nn.BatchNorm2d(128).cuda(gpu_device),
			    feats[6].cuda(gpu_device),
			    feats[7].cuda(gpu_device),
			    nn.BatchNorm2d(128).cuda(gpu_device),
			    feats[8].cuda(gpu_device),
			)
			self.CBR3_D = nn.Sequential(
			    feats[10].cuda(gpu_device),
			    nn.BatchNorm2d(256).cuda(gpu_device),
			    feats[11].cuda(gpu_device),
			    feats[12].cuda(gpu_device),
			    nn.BatchNorm2d(256).cuda(gpu_device),
			    feats[13].cuda(gpu_device),
			    feats[14].cuda(gpu_device),
			    nn.BatchNorm2d(256).cuda(gpu_device),
			    feats[15].cuda(gpu_device),
			)

			self.dropout3_d = nn.Dropout(p=0.5).cuda(gpu_device)

			self.CBR4_D = nn.Sequential(
			    feats[17].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats[18].cuda(gpu_device),
			    feats[19].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats[20].cuda(gpu_device),
			    feats[21].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats[22].cuda(gpu_device),
			)

			self.dropout4_d = nn.Dropout(p=0.5).cuda(gpu_device)

			self.CBR5_D = nn.Sequential(
			    feats[24].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats[25].cuda(gpu_device),
			    feats[26].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats[27].cuda(gpu_device),
			    feats[28].cuda(gpu_device),
			    nn.BatchNorm2d(512).cuda(gpu_device),
			    feats[29].cuda(gpu_device),
			) 
		if  rgb_d :
		
			####  RGB DECODER  ####
			self.CBR5_Dec = nn.Sequential (        
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Dropout(p=0.5).cuda(gpu_device),
			)

			self.CBR4_Dec = nn.Sequential (        
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(512, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(512, 256, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(256, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Dropout(p=0.5).cuda(gpu_device),
			)

			self.CBR3_Dec = nn.Sequential (        
			nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(256, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(256, 256, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(256, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(256,  128, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(128, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Dropout(p=0.5).cuda(gpu_device),
			)

			self.CBR2_Dec = nn.Sequential (
			nn.Conv2d(128, 128, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(128, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			nn.Conv2d(128, 64, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(64, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),
			)

			self.CBR1_Dec = nn.Sequential (                
			nn.Conv2d(64, 64, kernel_size=3, padding=1).cuda(gpu_device),
			nn.BatchNorm2d(64, momentum= batchNorm_momentum).cuda(gpu_device),
			nn.ReLU().cuda(gpu_device),        	
			nn.Conv2d(64, num_labels, kernel_size=3, padding=1).cuda(gpu_device),
			)

	def forward(self, rgb_inputs,depth_inputs):
		
		########  DEPTH ENCODER  ########

		# Stage 1
		x = self.conv11d(depth_inputs)
		x_1 = self.CBR1_D(x)
		x, id1_d = F.max_pool2d(x_1, kernel_size=2, stride=2, return_indices=True)
		
		# Stage 2
		x_2 = self.CBR2_D(x)
		x, id2_d = F.max_pool2d(x_2, kernel_size=2, stride=2, return_indices=True)

		# Stage 3
		x_3 = self.CBR3_D(x)
		x, id3_d = F.max_pool2d(x_3, kernel_size=2, stride=2, return_indices=True)
		x = self.dropout3_d(x)

		# Stage 4
		x_4 = self.CBR4_D(x)
		x, id4_d = F.max_pool2d(x_4, kernel_size=2, stride=2, return_indices=True)
		x = self.dropout4_d(x)

		# Stage 5
		x_5 = self.CBR5_D(x)

		########  RGB ENCODER  ########

		# Stage 1
		y = self.CBR1_RGB(rgb_inputs)
		y = torch.add(y,x_1)
		y, id1 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

		# Stage 2
		y = self.CBR2_RGB(y)
		y = torch.add(y,x_2)
		y, id2 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)

		# Stage 3
		y = self.CBR3_RGB(y)
		y = torch.add(y,x_3)
		y, id3 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
		y = self.dropout3(y)

		# Stage 4
		y = self.CBR4_RGB(y)
		y = torch.add(y,x_4)
		y, id4 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
		y = self.dropout4(y)

		# Stage 5
		y = self.CBR5_RGB(y)
		y = torch.add(y,x_5)
		y_size = y.size() 

		y, id5 = F.max_pool2d(y, kernel_size=2, stride=2, return_indices=True)
		y = self.dropout5(y)

		########  DECODER  ########

		# Stage 5 dec
		y = F.max_unpool2d(y, id5, kernel_size=2, stride=2, output_size=y_size)
		y = self.CBR5_Dec(y)

		# Stage 4 dec
		y = F.max_unpool2d(y, id4, kernel_size=2, stride=2)
		y = self.CBR4_Dec(y)

		# Stage 3 dec
		y = F.max_unpool2d(y, id3, kernel_size=2, stride=2)
		y = self.CBR3_Dec(y)

		# Stage 2 dec
		y = F.max_unpool2d(y, id2, kernel_size=2, stride=2)
		y = self.CBR2_Dec(y)

		# Stage 1 dec
		y = F.max_unpool2d(y, id1, kernel_size=2, stride=2)
		y = self.CBR1_Dec(y)

		return y
		
class SegmantationLoss(nn.Module):
	def __init__(self):
		super(SegmantationLoss, self).__init__()
		
		self.loss = nn.CrossEntropyLoss()

	def __call__(self, output, target, weights=None,pixel_average=True):
		if pixel_average:
			 self.loss(output, target, weights=weights,size_average=False) / target_mask.data.sum()
		else:
			return self.loss(output, target, weights=weights,size_average=False)
		
		
