#!/usr/bin/env python
#SBATCH --job-name=fusenet
#SBATCH --nodes=1
#SBATCH --cpus=4
#SBATCH --gres=gpu:1
#SBATCH --time="UNLIMITED"

import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
# from util.visualize_mask import *
from util.util import confusion_matrix, getScores
import numpy as np
import random
import torch
import cv2

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
	test_opt.phase = 'val'
	test_opt.batch_size = 1
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
	for epoch in range(train_opt.epoch_count, train_opt.niter + 1):
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

			iter_data_time = time.time()

		print('End of epoch %d / %d \t Time Taken: %d sec' %   (epoch, train_opt.niter, time.time() - epoch_start_time))
		model.update_learning_rate()
		if epoch > 100 and epoch % train_opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)

			model.eval()
			test_loss_iter = []
			gts = None
			preds = None
			epoch_iter = 0
			conf_mat = np.zeros((test_dataset.dataset.num_labels, test_dataset.dataset.num_labels), dtype=np.float)
			with torch.no_grad():
				for i, data in enumerate(test_dataset):
					model.set_input(data)
					model.forward()
					model.get_loss()
					epoch_iter += test_opt.batch_size
					gt = model.mask.cpu().int().numpy()
					_, pred = torch.max(model.output.data.cpu(), 1)
					pred = pred.float().detach().int().numpy()
					if test_dataset.dataset.name() == 'Scannetv2':
						gt = data["mask_fullsize"].cpu().int().numpy()[0]
						pred = cv2.resize(pred[0], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
					conf_mat += confusion_matrix(gt, pred, test_dataset.dataset.num_labels, ignore_label=test_dataset.dataset.ignore_label)
					# visualizer.display_current_results(model.get_current_visuals(), epoch, False)
					losses = model.get_current_losses()
					test_loss_iter.append(model.loss_segmentation)
					print('test epoch {0:}, iters: {1:}/{2:} '.format(epoch, epoch_iter, len(test_dataset) * test_opt.batch_size), end='\r')

			avg_test_loss = np.mean(test_loss_iter)
			glob,mean,iou = getScores(conf_mat)
			visualizer.print_current_scores(epoch, avg_test_loss, glob, mean, iou)
			visualizer.save_confusion_matrix(conf_mat, epoch)
