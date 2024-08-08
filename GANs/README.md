## 1. simple GAN
### Description:
Generative Adversarial Networks (GANs) as the name suggests is a framework to estimate generative models via an adversarial training process, in which two models (generator and discriminator) are trained simultaneously and the job of generator is to trick discriminator in identifying the data to be real or fake. As training proceeds, discriminator improves its ability to identify real/fake, and thus generator also improves its ability to produce (fake) images which seems more real-like. This is the main idea behind gans. Mathematically, this translates to sampling a **_z_** from a given prior **_p<sub>z</sub>_** and passing it through generator **_G_** to generate **_G(z)_** which follows a distribution similar to **_p<sub>r</sub>_**, where **_p<sub>r</sub>_** is the distribution of real data. The training aims to minimize the Jensen-Shannon Divergence (JSD) between the real and fake distribution, and the exact formulation of the losses corresponding to generator and discriminator is given in the paper (linked below). I have implemented the exact algo as mentioned in the original paper, with value of k=1 (Using JSD doesn't provide training stability, and as can be seen in the WGAN section, the reason of using Wasserstein distance was primarily the stability it provides while the trade-off being more computation because of slower convergence). I used [Celeb Faces](https://www.kaggle.com/datasets/farzadnekouei/50k-celebrity-faces-image-dataset?resource=download) dataset available on kaggle to train all gan variants and the results I obtained (training progress) is shown below in the form of a gif :).

Paper Link : [GAN paper](https://arxiv.org/abs/1406.2661)

<p align="center">
<img src="../assets/gif_simple_gan.gif" width="300"/>
</p>

To run the script, first clone the repo in your machine, and ``cd`` to ``GANs/simpleGAN/`` and run;
```sh
python train.py
```
Or to access the tensorboard (if you don't have it, install using ``pip install tensorboard``), run:
```sh
tensorboard --logdir=tensorboard/runs
```

## 2. DCGAN
### Description:

<p align="center">
<img src="../assets/gif_dcgan.gif" width="300"/>
</p>

## 3. Wasserstein GAN
### Description:

<p align="center">
<img src="../assets/gif_wgan.gif" width="300"/>
</p>

## 4. Wasserstein GAN (with gradient penalty)
### Description:

<p align="center">
<img src="../assets/gif_wgan_gp.gif" width="300"/>
</p>
