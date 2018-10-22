import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.visualize_mask import *
from util.util import confusion_matrix,getScores
import numpy as np

if __name__ == '__main__':
	train_opt = TrainOptions().parse()
	train_data_loader = CreateDataLoader(train_opt)
	train_dataset = train_data_loader.load_data()
	train_dataset_size = len(train_data_loader)
	print('#training images = %d' % train_dataset_size)

        test_opt = TrainOptions().parse()
        test_opt.phase = 'test'
        test_opt.num_threads = 1
        test_data_loader = CreateDataLoader(test_opt)
        test_dataset = test_data_loader.load_data()
	test_dataset_size = len(test_data_loader)
	print('#test images = %d' % test_dataset_size)

        model = create_model(train_opt)
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

			if total_steps % train_opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %  (epoch, total_steps))
				model.save_networks('latest')

			iter_data_time = time.time()
		if epoch % train_opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' %   (epoch, train_opt.niter + train_opt.niter_decay, time.time() - epoch_start_time))
		model.update_learning_rate()
                if epoch % 5 == 0:
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
                            epoch_iter =+ test_opt.batch_size
                            gt = model.mask.cpu().int().numpy()
                            _, pred = torch.max(model.output.data.cpu(), 1)
                            pred = pred.float().detach().int().numpy()
                            if gts is None:
                                gts = gt
                                preds = pred
                            else :
                                gts = np.concatenate((gts, gt), axis=0)
                                preds = np.concatenate((preds, pred), axis=0)
                            visualizer.display_current_results(model.get_current_visuals(), epoch, False)
                            losses = model.get_current_losses()
                            test_loss_iter.append(model.loss_segmentation)

                        avg_test_loss = np.mean(test_loss_iter)
                        print ('Epoch {} test loss {}: '.format(epoch, avg_test_loss))
                        conf_mat = confusion_matrix(gts,preds,40,ignore_label=0)
                        glob,mean,iou = getScores(conf_mat)
                        print ('Epoch {} glob acc : {}, mean acc : {}, IoU : {}'.format(epoch,glob,mean,iou))
