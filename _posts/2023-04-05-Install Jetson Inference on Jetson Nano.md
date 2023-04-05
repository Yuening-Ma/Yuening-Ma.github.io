---
title: Install Jetson Inference on Jetson Nano
date: 2023-04-05 21:26:05 +0800
categories: [Develop Environment, Jetson Platform]
tags: [ai, jetson, pytorch, dev-environment]     # TAG names should always be lowercase
toc: true
mermaid: true
---

# Why I choose Jetson Inference

I'm making a robot car recently, and after trying various Linux developmemt boards from different manufacturers (e.g. Raspberry Pi, Allwinner H6, RK3399 and RK3588), I chose Jetson Nano 4GB as my robot car's master control. 

According to Pytorch office post ([Running PyTorch Models on Jetson Nano](https://pytorch.org/blog/running-pytorch-models-on-jetson-nano/)), on Jetson Nano, there are THREE approaches to run deep learning models on Jetson Nano with Pytorch:

* [Jetson Inference](https://github.com/dusty-nv/jetson-inference), which is a higher-level NVIDIA API and able to utilize Nvidia GPU efficiently and neatly. Besides, you can easily accomplish thansfer learning based on pytorch pre-trained models and deploy the models on Jetson Nano using Jetson Inference.

* [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/), which is an SDK for high-performance inference from NVIDIA but require certain model format converting.

* [PyTorch](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) with the direct PyTorch API torch.nn for inference.

For developers who are willing to use pytorh pre-trained models and not like to deal with GPU developing environment (cuda, cudnn, TendorRT and so on), Jetson Inference on Jetson official system is a nice choice.

Below is Jetson Inference installation instruction according to **Pytorch**, **Nvidia** and **Jetson Iinference** official materials and some **DEBUGGING PROCESS**.

# Setting up Jetson Nano

This part is according to the [Pytorch office post](https://pytorch.org/blog/running-pytorch-models-on-jetson-nano/) mentioned above.

## Initial works like flashing the system

Simply follow this [instruction](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit) to do some initial works like flash the system. After this procedure, your Jetson Nano will be running a Ubuntu equipped with all Nvidia gpu accelerated computing related things (e.g. TensorRT, CUDA, cudnn).

In my case, the up-to-date Ubuntu version for Jetson Nano is `18.04` and the JetPack version is `4.6.3`.

![set-up-jetson-nano](/_posts/2023-04-05-imgs/set-up-jetson-nano.png)

## Check GPU and CUDA infomation

To check the GPU status on Nano, run the following commands:

```bash
sudo pip3 install jetson-stats
sudo jtop
```
![jtop](/_posts/2023-04-05-imgs/jtop.png)

You can also see the installed CUDA version:

```
$ ls -lt /usr/local
lrwxrwxrwx  1 root root   22 Aug  2 01:47 cuda -> /etc/alternatives/cuda
lrwxrwxrwx  1 root root   25 Aug  2 01:47 cuda-10 -> /etc/alternatives/cuda-10
drwxr-xr-x 12 root root 4096 Aug  2 01:47 cuda-10.2
```

![check-CUDA-version](/_posts/2023-04-05-imgs/check-CUDA-version.png)


# Clone the Github repository

The procedures below is according mainly to [Building the Project from Source](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md) on [Jetson Inference Github repo](https://github.com/dusty-nv/jetson-inference).

Your can [Running the Docker Container](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md) rather than building from source and use jetson inference in dorker, in which condition you can skip the part remaining. 

However, if you prefer using the APIs in your local environment, building from source is necessary.

First, make sure git and cmake are installed:

```
$ sudo apt update
$ sudo apt install git cmake
```

Navigate to a folder of your choosing on the Jetson, and clone the jetson-inference project:

```
$ git clone --recursive https://github.com/dusty-nv/jetson-inference
```

The `--recursive` param is necessary for getting all submodules needed.

# Install Python dependencies

If you want the project to create bindings for Python 3.6, install these packages:

```
$ sudo apt install libpython3-dev python3-numpy
```

> Tips: on Ubuntu arm platform, using `sudo apt install python3-xxx` instead of `pip3 install xxx` will be faster sometimes, as there may not be pre-compiled wheel packages in pip arm source, and the pip package compiling can be time-comsuming.

# Download models

As cmake step may encounter a bug following official instruction, we skip this step temporarily and make the other stuffs down first.


```
$ cd jetson-inference/tools
$ ./download-models.sh
```

You can select the models you want, or run the tool again later to download more models another time.

For now, the default models is enough.

![download-models](/_posts/2023-04-05-imgs/download-models.jpg)


# Install Pytorch (optional)

This step is optional, and if you don't wish to do the transfer learning steps on Jetson Nano, you don't need to install PyTorch and can skip this step.

```
$ cd ../build
$ ./install-pytorch.sh
```

Select the PyTorch package versions for Python 2.7 and/or Python 3.6 that you want installed and hit Enter to continue.

![pytorch-installer](/_posts/2023-04-05-imgs/pytorch-installer.jpg)

You can also run this tool again later if you decide that you want to install PyTorch at another time.

# Install torchvision (optional, not completed yet)

In last step, torchvision will be installed automatedly. However, in my case, run `import torchvision` will raise error.

According to the official reply in [this post on Nvidia developer forum](https://forums.developer.nvidia.com/t/installing-torchvision/245286):

> TorchVision needs to build from the source on Jetson. You can find the steps and corresponding version below: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 > Installation > Torchvision

So open [the post mentioned above](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048), and select Instructions > Installation > Torchvision in the top floor, get these procedures:

```
$ sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
$ git clone --branch <version> https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
$ cd torchvision
$ export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version  
$ python3 setup.py install --user
$ cd ../  # attempting to load torchvision from build dir will result in import error
$ pip install 'pillow<7' # always needed for Python 2.7, not needed torchvision v0.5.0+ with Python 3.6
```

`python3 setup.py install --user` may raise an error:

```
Error: 'libavformat/avformat.h: No such file or directory.'
```

Some posts give this solution: `sudo apt-get install libavformat53 libavformat-dev libavcodec53`.

However, for me, `ffmpeg` have to be reinstalled from source, so that some libraries related like libavformat will be correctly set up in my system.

To do so, follow the [ffmpeg official installation instruction](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu).

After the ffmpeg installation steps, there would be two directories in your home path:
* ffmpeg_build
* bin
According the code in `torchvision setup.py file`, move the whole `bin` directory to `ffmpeg_build/`, and add `ffmpeg_build` to PATH:
```
export PATH=$PATH:/home/xxx/ffmpeg_build
```
Your can save this in your `~/.bashrc` file.

Now you can run `python3 setup.py install --user`.

:confused::confused::confused: TODO: error about ['ninja', '-v'] :confused::confused::confused: 


# Debug for CMake error 








# Compiling the Project

Ensuring you are in `jetson-inference/build/` directory, and run:

```
$ make -j4
$ sudo make install
$ sudo ldconfig
```

And congradulations! You have successfully install Jetson Inference on your Jetson Nano.

Now you can run some examples according to [this doc](https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md#detecting-objects-from-the-command-line):



