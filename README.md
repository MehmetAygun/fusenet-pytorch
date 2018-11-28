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

## FuseNet train/test

### visdom visualization
To view training errors and loss plots, set `--display_id 1`, run `python -m visdom.server` and click the URL http://localhost:8097. Checkpoints are saved under `./checkpoints/sunrgbd/`.

### train
```bash
python train.py --dataroot datasets/sunrgbd --model fusenet --dataset sunrgbd --name sunrgbd --no_html --batch_size 4 --num_threads 8
```

### test
```bash
python test.py --dataroot datasets/sunrgbd --model fusenet --dataset sunrgbd --name sunrgbd --gpu_ids 0 --epoch 305
```

### Optimization scheme and loss weights
* We use the training scheme defined in FuseNet
* Loss is weighted for SUNRGBD dataset
* Learning rate is set to 0.01 for NYUv2 dataset
* Results can be improved with a hyper-parameter search

<table>
<tr>
<td colspan=1> <b>Dataset <td colspan=3> <b>FuseNet-SF5 (CAFFE) <td colspan=3> <b>FuseNet-SF5
<tr>
<td> <td> overall <td> mean <td> iou <td> overall <td> mean <td> iou
<tr>
<td> <b>sunrgbd <td> 76.30 <td> 48.30 <td> 37.30
<tr>
<td> <a href="https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/400_net_FuseNet.pth"> <b>nyuv2 </a> <td> 66.00 <td> 43.40 <td> 32.70 <td>  68.76 <td> 46.42 <td> 35.48
<tr>
<td> <b>scannet <td> -- <td> -- <td> --
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
