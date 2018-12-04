import numpy as np
import h5py
import scipy.io as sio
import cv2
import glob
from PIL import Image

def calc_scannetv2(data_root,n_class):
    masks = []
    size = (320,240)
    with open('./datasets/scannet/scannetv2_{}.txt'.format('train')) as f:
        scans = f.readlines()
        scans = [x.strip() for x in scans]
    for scan in scans:
        ms = glob.glob("{}/{}/label/*.png".format(data_root, scan))
        masks.extend(ms)
    mask_numpy = []
    num_images = np.zeros((n_class))
    for index in range(len(masks)):
        mask = np.array(Image.open(masks[index]))
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        num_images[np.unique(mask)] += 1
        mask_numpy.append(mask)

    mask_numpy = np.array(mask_numpy)
    counts = np.array(np.unique(mask_numpy, return_counts=True)).T
    freqs = counts [:,1] / num_images
    weights = np.median(freqs) / freqs;
    np.savetxt('./datasets/scannet/scannetv2_weigths.txt',weights)

def calc_weigths(dataset,data_root):
	if dataset == "scannetv2":
		n_class = 41
		calc_scannetv2(data_root,n_class)
	else:
		print ("Dataset {} is not implemented".format(dataset))

def main():
    data_root = '/usr/data/cvpr_shared/common_datasets/scannet/tasks/scannet_frames_25k'
    calc_weigths("scannetv2",data_root)

if __name__ == '__main__':
	main()
