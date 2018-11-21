<!-- <img src='imgs/horse2zebra.gif' align="right" width=384> -->
<!-- <br><br><br> -->

# FuseNet implementation in PyTorch

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
* Results can be improved with a hyper-parameter search

| Dataset       | FuseNet (CAFFE) | FuseNet |
| ------------- |:----:| :----: |

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
