## Description
This repo includes my block-by-block implementations from scratch of several CNNs and GANs using PyTorch. The goal of this project is to understand the intricacies of development of deep learning models using PyTorch and theoretical details behind each model. Language models will be added later with a detailed documentation ``WIP``.

## Directory Structure
```
.
├── README.md
│
├── assets/                          # to add in readme
│
├── CNNs/
│   ├── lenet5.py                    # architecture of LeNet-5
│   ├── vggnet.py                    # architectures of VGG11, VGG13, VGG16, VGG19
│   ├── googlenet.py                 # architecture of GoogLeNet (inception net)
│   └── resnet.py                    # architectures of ResNet-50, ResNet-101, ResNet-152
│
└── GANs/
    ├── README.md
    │
    ├── simpleGAN/                   # scripts to model and train GAN
    │   ├── tensorboard/runs/
    │   ├── model.py                 
    │   └── train.py                 
    │   
    ├── DCGAN/                       # scripts to model and train DCGAN
    ├── WGAN/                        # scripts to model and train WGAN
    └── WGAN-GP/                     # scripts to model and train WGAN with gradient penalty
```
