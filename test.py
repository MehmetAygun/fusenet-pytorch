import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    model.eval()
    test_loss_iter = []
    with torch.no_grad(): 
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break
            model.set_input(data)
            model.forward()
            model.get_loss()
            test_loss_iter.append(model.loss_segmentation)

    # save the website
    webpage.save()
    avg_test_loss = np.mean(test_loss_iter)
    print avg_test_loss


