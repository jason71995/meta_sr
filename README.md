# Meta-SR in Keras and Tensorflow 2.0

## Introduction

Implementation of [(2019) Meta-SR: A Magnification-Arbitrary Network for Super-Resolution](https://arxiv.org/abs/1903.00875).

| Nearest  | Bicubic | Meta-SR | Real |
| :---: | :---: | :---: | :---: |
| ![alt text](https://i.imgur.com/9jPrBn4.png "Nearest") | ![alt text](https://i.imgur.com/cOsvDPt.png "Bicubic") |![alt text](https://i.imgur.com/pAKiBDm.png "Meta-SR") |![alt text](https://i.imgur.com/jeJNxWV.png "HR") |
| ![alt text](https://i.imgur.com/qSYbCQw.png "Nearest") | ![alt text](https://i.imgur.com/8qChRXH.png "Bicubic") |![alt text](https://i.imgur.com/eDR268C.png "Meta-SR") |![alt text](https://i.imgur.com/8WDixsQ.png "HR") |

## Attention

- The paper using **Matlab** bicubic resize for LR image, but here using **pillow** bicubic resize.
- Pretrained model train on x1.0 to x4.0, if you want test over x4.0 better to retrain the model.

## Data

- Download training data from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- Get pretrained model from [Google Drive](https://drive.google.com/file/d/10r2tK9Y74Fx--u_TFI7HcTodyP3pjVaF/view?usp=sharing)

## Environment
```
python==3.6
tensorflow==2.0
```

## How to use

#### Train
```
python train.py
```
#### Predict
```
python predict.py --image test.png --model weights.h5 --scale 4.0
```

## To Do
- Add MetaRDN

## Reference
- Official implementation in pytorch. [link](https://github.com/XuecaiHu/Meta-SR-Pytorch)