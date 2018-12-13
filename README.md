<!-- <img src='imgs/horse2zebra.gif' align="right" width=384> -->
<!-- <br><br><br> -->

# [FuseNet](https://github.com/tum-vision/fusenet) implementation in PyTorch

This is the PyTorch implementation for FuseNet, developed based on [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) code.

## Prerequisites
- Linux
- Python 3.7.0
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch 0.4.1.post2 and dependencies from http://pytorch.org
- Clone this repo:
```bash
git clone https://github.com/MehmetAygun/fusenet-pytorch
cd fusenet-pytorch
pip install -r requirements.txt
```
## Dataset preparation
### sunrgbd dataset
- Download and untar the [preprocessed sunrgbd](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/sun/sunrgbd.tar.gz) dataset under ```/datasets/sunrgbd```

### nyuv2 dataset
- Download the dataset and create the training set
```bash
cd datasets
sh download_nyuv2.sh
python create_training_set.py
```
### scannetv2 dataset
- Download the ```scannet_frames_25k``` and ```scannet_frames_test``` under ```/datasets/scannet/tasks/```

## FuseNet train/test

### visdom visualization
- To view training errors and loss plots, set `--display_id 1`, run `python -m visdom.server` and click the URL http://localhost:8097
- Checkpoints are saved under `./checkpoints/sunrgbd/`

### train & test on sunrgbd
```bash
python train.py --dataroot datasets/sunrgbd --dataset sunrgbd --name sunrgbd

python test.py --dataroot datasets/sunrgbd --dataset sunrgbd --name sunrgbd --epoch 400
```

### train & test on nyuv2
```bash
python train.py --dataroot datasets/nyuv2 --dataset nyuv2 --name nyuv2

python test.py --dataroot datasets/nyuv2 --dataset nyuv2 --name nyuv2 --epoch 400
```

### train & val & test on scannetv2
```bash
python train.py --dataroot datasets/scannet/tasks/scannet_frames_25k --dataset scannetv2 \
                --name scannetv2

python test.py --dataroot datasets/scannet/tasks/scannet_frames_25k --dataset scannetv2 \
               --name scannetv2 --epoch 260 --phase val

python test.py --dataroot datasets/scannet/tasks/scannet_frames_test --dataset scannetv2 \
               --name scannetv2 --epoch 260 --phase test
```

## Results
* We use the training scheme defined in FuseNet
* Loss is weighted for SUNRGBD dataset
* Learning rate is set to 0.01 for NYUv2 dataset
* Results can be improved with a hyper-parameter search
* Results on the scannetv2-test (w/o class-weighted loss) can be found [here](http://kaldir.vc.in.tum.de/scannet_benchmark/result_details?id=67)

<table>
<tr>
<td colspan=1> <b>Dataset <td colspan=3> <b>FuseNet-SF5 (CAFFE) <td colspan=3> <b>FuseNet-SF5
<tr>
<td> <td> overall <td> mean <td> iou <td> overall <td> mean <td> iou
<tr>
<td> <a href="https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/sun/400_net_FuseNet.pth"> <b>sunrgbd </a> <td> 76.30 <td> 48.30 <td> 37.30 <td> 75.41 <td> 46.48 <td> 35.69
<tr>
<td> <a href="https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/400_net_FuseNet.pth"> <b>nyuv2 </a> <td> 66.00 <td> 43.40 <td> 32.70 <td>  68.76 <td> 46.42 <td> 35.48
<tr>
<td> <a href="https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/scannet/260_net_FuseNet.pth"> <b>scannetv2-val </a> <td> -- <td> -- <td> -- <td> 76.32 <td> 55.84 <td> 44.12
<tr>
<td> <a href="https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/scannet/380_net_FuseNet.pth">
<b>scannetv2-cls_weighted-val </a> <td> -- <td> -- <td> -- <td> 76.26 <td> 55.74 <td> 44.40
</table>

## Citation
```
@inproceedings{hazirbas16fusenet,
  Title                    = {{FuseNet}: Incorporating Depth into Semantic Segmentation via Fusion-Based CNN Architecture},
  Author                   = {Hazirbas, Caner and Ma, Lingni and Domokos, Csaba and Cremers, Daniel},
  Booktitle                = {Asian Conference on Computer Vision ({ACCV})},
  Year                     = {2016},
  Doi                      = {10.1007/978-3-319-54181-5_14},
  Url                      = {https://github.com/tum-vision/fusenet}
}
```
## Acknowledgments
Code is inspired by [pytorch-CycleGAN-and-pix2pix]((https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)).
