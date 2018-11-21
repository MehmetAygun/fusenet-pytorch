import numpy as np
import h5py
import scipy.io as sio
import cv2
import zipfile

def create_nyu2():

		print ("Loading mat file ...")

		f = h5py.File('nyu_depth_v2_labeled.mat')
		rgb_images = np.array([x for x in f["images"]])
		depth_images = np.array([x for x in f["depths"]])
		labels= np.array([x for x in f["labels"]])

		mapping =  sio.loadmat("nyuv2_40class_mapping.mat")["mapping"][0]

		print ("Mapping 894 class to 40 class...")

		for i,label in enumerate(labels):
			for j,row in enumerate(label):
				for k,value in enumerate(row):
					labels[i][j][k] = mapping[value-1] - 1

		print ("Resizing and scaling images and labels...")

		rgbs=[]
		depths=[]
		masks=[]

		max_depth,min_depth = np.max(depth_images),np.min(depth_images)


		for rgb_image in rgb_images:
			img = np.transpose(rgb_image,(2,1,0))
			img = cv2.resize(img,(320,240),interpolation=cv2.INTER_LINEAR)
			rgbs.append(img)

		for depth_image in depth_images:
			depth_image = np.transpose(depth_image,(1,0))
			depth_image = (depth_image - min_depth) / (1.*(max_depth-min_depth))*255
			depth_image = cv2.resize(depth_image,(320,240),interpolation=cv2.INTER_NEAREST)
			depth_image = depth_image.astype(int)
			depths.append(np.array(depth_image,dtype=np.uint8))

		for label in labels:
			label = np.transpose(label,(1,0))
			label = cv2.resize(label,(320,240),interpolation=cv2.INTER_NEAREST)
			masks.append(label)

		d = { "rgb_images":rgbs, "depth_images":depths, "masks":masks}

		print ("Saving images and labels...")

		np.save("nyu2.npy", d)

		print ("Finished !")

def create_scannet(path='/usr/data/cvpr_shared/common_datasets/scannet/scans',mode='train'):

	scan_paths = [x[0] for x in os.walk("/usr/data/cvpr_shared/common_datasets/scannet/scans/")][1:]
	for scan_path in scan_paths:
		scan_name = scan_path.split("/")[-1]
		if not os.path.exists("{}/{}_2d-label".format(scan_path,scan_name)):
			zip_ref = zipfile.ZipFile("{}/{}_2d-label.zip".format(scan_path,scan_name), 'r')
			zip_ref.extractall("{}/{}_2d-label".format(scan_path,scan_name))
			zip_ref.close()
		if not os.path.exists("{}/{}_2d-instance".format(scan_path,scan_name)):
			zip_ref = zipfile.ZipFile("{}/{}_2d-instance.zip".format(scan_path,scan_name), 'r')
			zip_ref.extractall("{}/{}_2d-instance".format(scan_path,scan_name))
			zip_ref.close()


	print ("Finished !")

def create_dataset(dataset):

	if dataset == "nyu2":
		create_nyu2()
	elif dataset == "scannet":
		create_scannet()
	else:
		print ("Dataset {} is not implemented".format(dataset))

def main():
	create_dataset("nyu2")

if __name__ == '__main__':
	main()
