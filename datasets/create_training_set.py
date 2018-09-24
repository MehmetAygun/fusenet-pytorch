import numpy as np
import h5py
import scipy.io as sio

def create_dataset(dataset):

	if dataset == "nyu2":

		print "Loading mat file ..."

		f = h5py.File('nyu_depth_v2_labeled.mat')
		rgb_images = np.array([x for x in f["images"]])
		depth_images = np.array([x for x in f["depths"]]) 
		labels= np.array([x for x in f["labels"]])  

		mapping =  sio.loadmat("nyuv2_40class_mapping.mat")["mapping"][0]
		
		print "Mapping 894 class to 40 class..."

		for i,label in enumerate(labels):
			for j,row in enumerate(label):
				for k,value in enumerate(row):
					labels[i][j][k] = mapping[value-1]

		d = { "rgb_images":rgb_images, "depth_imahes":depth_images, "labels":labels}

		print "Saving images and labels..."

		np.save("nyu2.npy", d)

		print "Finished !"

	else:
		print "Dataset {} is not implemented".format(dataset)

def main():
	create_dataset("nyu2")

if __name__ == '__main__':
	main()

