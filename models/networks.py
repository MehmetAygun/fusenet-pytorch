import torch
import torch.nn as nn
from torch.nn import init
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

def define_encoder(input_nc, output_nc, ngf, netE, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if netE == 'vgg_16_rgb_encoder':
		net = VggGenerator(rgb=True)
	elif netE == 'vgg_16_depth_encoder':
		net = VggGenerator(rgb=False)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % netE)

	return init_net(net, init_type, init_gain, gpu_ids)

def define_dencoder(input_nc, output_nc, ngf, netE, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
	 net = None
	 norm_layer = get_norm_layer(norm_type=norm)



def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if netG == 'resnet_9blocks':
		net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif netG == 'resnet_6blocks':
		net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
	elif netG == 'unet_128':
		net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
	elif netG == 'unet_256':
		net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
	return init_net(net, init_type, init_gain, gpu_ids)



##############################################################################
# Classes
##############################################################################
class VggGenerator(nn.Module):
	def __init__(self, rgb=True):
		super(VggGenerator, self).__init__()
		self.rgb = rgb

		self.model = nn.Sequential(*model)

	def forward(self, input):
		return self.model(input)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
	def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		if use_lsgan:
			self.loss = nn.MSELoss()
		else:
			self.loss = nn.BCELoss()

	def get_target_tensor(self, input, target_is_real):
		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(input)

	def __call__(self, input, target_is_real):
		target_tensor = self.get_target_tensor(input, target_is_real)
		return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
		assert(n_blocks >= 0)
		super(ResnetGenerator, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.ngf = ngf
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
						   bias=use_bias),
				 norm_layer(ngf),
				 nn.ReLU(True)]

		n_downsampling = 2
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
								stride=2, padding=1, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]

		mult = 2**n_downsampling
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		for i in range(n_downsampling):
			mult = 2**(n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]

		self.model = nn.Sequential(*model)

	def forward(self, input):
		return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim),
					   nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
					   norm_layer(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out
