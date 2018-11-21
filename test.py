import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_images
from util.util import confusion_matrix, getScores
from util import html
import torch
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    # opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt, dataset.dataset)
    model.setup(opt)
    model.eval()
    visualizer = Visualizer(opt)

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    test_loss_iter = []
    epoch_iter = 0
    conf_mat = np.zeros((dataset.dataset.num_labels, dataset.dataset.num_labels), dtype=np.float)
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.forward()
            model.get_loss()
            epoch_iter += opt.batch_size
            gt = model.mask.cpu().int().numpy()
            _, pred = torch.max(model.output.data.cpu(), 1)
            pred = pred.float().detach().int().numpy()
            conf_mat += confusion_matrix(gt, pred, dataset.dataset.num_labels, ignore_label=dataset.dataset.ignore_label)
            save_images(webpage, model.get_current_visuals(), model.get_image_paths())
            losses = model.get_current_losses()
            test_loss_iter.append(model.loss_segmentation)
            print('Epoch {0:}, iters: {1:}/{2:}, loss: {3:.3f} '.format(opt.epoch,
                                                                        epoch_iter,
                                                                        len(dataset) * opt.batch_size,
                                                                        test_loss_iter[-1]), end='\r')

        print()
        avg_test_loss = np.mean(test_loss_iter)
        print ('Epoch {0:} test loss: {1:.3f} '.format(opt.epoch, avg_test_loss))
        glob,mean,iou = getScores(conf_mat)
        print ('Epoch {0:} glob acc : {1:.2f}, mean acc : {2:.2f}, IoU : {3:.2f}'.format(opt.epoch, glob, mean, iou))
        print('Confusim matrix is saved to ' + visualizer.conf_mat_name)
        visualizer.save_confusion_matrix(conf_mat, opt.epoch)

    # save the website
    webpage.save()
