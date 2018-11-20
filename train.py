#!/usr/bin/env python
#SBATCH --job-name=fusenet
#SBATCH --nodes=1
#SBATCH --cpus=10
#SBATCH --gres=gpu:1
#SBATCH --mem=12GB
#SBATCH --time="UNLIMITED"

import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
# from util.visualize_mask import *
from util.util import confusion_matrix,getScores
import numpy as np
import random
import torch

if __name__ == '__main__':
	train_opt = TrainOptions().parse()

	np.random.seed(train_opt.seed)
	random.seed(train_opt.seed)
	torch.manual_seed(train_opt.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.cuda.manual_seed(train_opt.seed)

	train_data_loader = CreateDataLoader(train_opt)
	train_dataset = train_data_loader.load_data()
	train_dataset_size = len(train_data_loader)
	print('#training images = %d' % train_dataset_size)

	test_opt = TrainOptions().parse()
	test_opt.phase = 'test'
	test_opt.num_threads = 1
	test_opt.serial_batches = True
	test_opt.no_flip = True
	test_opt.display_id = -1
	test_data_loader = CreateDataLoader(test_opt)
	test_dataset = test_data_loader.load_data()
	test_dataset_size = len(test_data_loader)
	print('#test images = %d' % test_dataset_size)

	model = create_model(train_opt, train_dataset.dataset)
	model.setup(train_opt)
	visualizer = Visualizer(train_opt)
	total_steps = 0
	for epoch in range(train_opt.epoch_count, train_opt.niter + train_opt.niter_decay + 1):
		model.train()
		epoch_start_time = time.time()
		iter_data_time = time.time()
		epoch_iter = 0
		for i, data in enumerate(train_dataset):
			iter_start_time = time.time()
			if total_steps % train_opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time
			visualizer.reset()
			total_steps += train_opt.batch_size
			epoch_iter += train_opt.batch_size
			model.set_input(data)
			model.optimize_parameters()

			if total_steps % train_opt.display_freq == 0:
				save_result = total_steps % train_opt.update_html_freq == 0
				visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

			if total_steps % train_opt.print_freq == 0:
				losses = model.get_current_losses()
				t = (time.time() - iter_start_time) / train_opt.batch_size
				visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
				if train_opt.display_id > 0:
					visualizer.plot_current_losses(epoch,
                                                float(epoch_iter) / train_dataset_size, train_opt, losses)

			# if total_steps % train_opt.save_latest_freq == 0:
			# 	print('saving the latest model (epoch %d, total_steps %d)' %  (epoch, total_steps))
			# 	model.save_networks('latest')

			iter_data_time = time.time()
		if epoch % train_opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' %   (epoch, train_opt.niter + train_opt.niter_decay, time.time() - epoch_start_time))
		model.update_learning_rate()
		if epoch > train_opt.niter and epoch % 5 == 0:
			model.eval()
			test_loss_iter = []
			gts = None
			preds = None
			epoch_iter = 0
			with torch.no_grad():
				for i, data in enumerate(test_dataset):
					model.set_input(data)
					model.forward()
					model.get_loss()
					epoch_iter += test_opt.batch_size
					gt = model.mask.cpu().int().numpy()
					_, pred = torch.max(model.output.data.cpu(), 1)
					pred = pred.float().detach().int().numpy()
					if gts is None:
					    gts = gt
					    preds = pred
					else :
					    gts = np.concatenate((gts, gt), axis=0)
					    preds = np.concatenate((preds, pred), axis=0)
					# visualizer.display_current_results(model.get_current_visuals(), epoch, False)
					losses = model.get_current_losses()
					test_loss_iter.append(model.loss_segmentation)
					print('test epoch {0:}, iters: {1:} '.format(epoch, epoch_iter), end='\r')

				avg_test_loss = np.mean(test_loss_iter)
				print ('Epoch {0:} test loss: {1:.3f} '.format(epoch, avg_test_loss))
				conf_mat = confusion_matrix(gts, preds, train_dataset.dataset.num_labels, ignore_label=train_dataset.dataset.ignore_label)
				glob,mean,iou = getScores(conf_mat)
				print ('Epoch {0:} glob acc : {1:.2f}, mean acc : {2:.2f}, IoU : {3:.2f}'.format(epoch, glob, mean, iou))
				visualizer.save_confusion_matrix(conf_mat, epoch)
