
Official implementation of CADNet: Change-Aware Siamese Network for Surface Defects Segmentation under Complex Background
[arxiv](https://arxiv.org/pdf/2409.00589)

   <div align=center>
   <img src="https://github.com/HATFormer/CADNet/blob/main/image/SeaNet.png">
   </div>

## Installation
### Install pytorch
```bash
conda create -n cadnet python=3.8
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
```
### Install MMSegmentation
```bash
pip install -U openmim
mim install mmengine
pip install mmcv-full
pip install -V -e .
```
### Install basic libraries
```bash
pip install scikit-learn
pip install scikit-image
pip install catboost
```
## Datasets used in the paper
Please download and put the following datasets in 'data' directory
[SynLCD](https://pan.baidu.com/s/165cP18FL2qxWz1ljEzUZDg?pwd=aicw)(password: aicw)
[PKU-PCB](https://pan.baidu.com/s/1OFsBKF4MY9eOTkTBW7eWtg?pwd=l1l6)(password: l1l6)

## For testing
Please download the trained models [here](https://pan.baidu.com/s/19cxdhoafK-g-M-0t6k0Adg?pwd=f6wp)(password:f6wp) and put them in the work_dirs directory.
```bash
python tools/test.py work_dirs/CADNet/mixed/CADNet.py work_dirs/CADNet/mixed/iter_126000.pth --eval mIoU mFscore
python tools/test.py work_dirs/CADNet_pcb/CADNet.py work_dirs/CADNet_pcb/iter_46980.pth --eval mIoU mFscore
```
## For training
```bash
python tools/train.py config_file.py
For example:
python tools/train.py work_dirs/CADNet/mixed/CADNet.py 
```
## Citation
```bash
@article{liu2024change,
  title={Change-Aware Siamese Network for Surface Defects Segmentation under Complex Background},
  author={Liu, Biyuan and Chen, Huaixin and Zhan, Huiyao and Luo, Sijie and Huang, Zhou},
  journal={arXiv preprint arXiv:2409.00589},
  year={2024}
}
```

