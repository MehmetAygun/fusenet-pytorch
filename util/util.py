from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))* 255.0
    return image_numpy.astype(imtype)


def tensor2labelim(label_tensor, impalette, imtype=np.uint8):
    if len(label_tensor.shape) == 4:
        _, label_tensor = torch.max(label_tensor.data.cpu(), 1)

    label_numpy = label_tensor[0].cpu().float().detach().numpy()
    label_image = Image.fromarray(label_numpy.astype(np.uint8))
    label_image = label_image.convert("P")
    label_image.putpalette(impalette)
    label_image = label_image.convert("RGB")
    return np.array(label_image).astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def confusion_matrix(x , y, n, ignore_label=None, mask=None):
        if mask is None:
            mask = np.ones_like(x) == 1
        k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
        return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def getScores(conf_matrix):
        if conf_matrix.sum() == 0:
            return 0, 0, 0
        with np.errstate(divide='ignore',invalid='ignore'):
            overall = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
            perclass = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
            IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        return overall * 100., np.nanmean(perclass) * 100., np.nanmean(IU) * 100.
