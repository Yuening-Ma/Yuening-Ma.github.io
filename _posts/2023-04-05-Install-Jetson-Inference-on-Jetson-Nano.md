---
title: Install Jetson Inference on Jetson Nano
date: 2023-04-05 21:26:05 +0800
categories: [Develop Environment, Jetson Platform]
tags: [ai, jetson, pytorch, dev-environment]     # TAG names should always be lowercase
toc: true
mermaid: true
---

# Why I choose Jetson Inference

![deep-vision-header](/assets/img/for_posts/2023-04-05/deep-vision-header.jpg)
> Image source: https://github.com/dusty-nv/jetson-inference

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

![set-up-jetson-nano](/assets/img/for_posts/2023-04-05/set-up-jetson-nano.png)

## Check GPU and CUDA infomation

To check the GPU status on Nano, run the following commands:

```bash
sudo pip3 install jetson-stats
sudo jtop
```
![jtop](/assets/img/for_posts/2023-04-05/jtop.png)

You can also see the installed CUDA version:

```bash
$ ls -lt /usr/local
```
```
lrwxrwxrwx  1 root root   22 Aug  2 01:47 cuda -> /etc/alternatives/cuda
lrwxrwxrwx  1 root root   25 Aug  2 01:47 cuda-10 -> /etc/alternatives/cuda-10
drwxr-xr-x 12 root root 4096 Aug  2 01:47 cuda-10.2
```

![check-CUDA-version](/assets/img/for_posts/2023-04-05/check-CUDA-version.png)


# Clone the Github repository

The procedures below is according mainly to [Building the Project from Source](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md) on [Jetson Inference Github repo](https://github.com/dusty-nv/jetson-inference).

Your can [Running the Docker Container](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md) rather than building from source and use jetson inference in dorker, in which condition you can skip the part remaining. 

However, if you prefer using the APIs in your local environment, building from source is necessary.

First, make sure git and cmake are installed:

```bash
sudo apt update
sudo apt install git cmake
```

Navigate to a folder of your choosing on the Jetson, and clone the jetson-inference project:

```bash
git clone --recursive https://github.com/dusty-nv/jetson-inference
```

The `--recursive` param is necessary for getting all submodules needed.

# Install Python dependencies

If you want the project to create bindings for Python 3.6, install these packages:

```bash
sudo apt install libpython3-dev python3-numpy
```

> Tips: on Ubuntu arm platform, using `sudo apt install python3-xxx` instead of `pip3 install xxx` will be faster sometimes, as there may not be pre-compiled wheel packages in pip arm source, and the pip package compiling can be time-comsuming.

# Download models

As cmake step may encounter a bug following official instruction, we skip this step temporarily and make the other stuffs down first.


```bash
cd jetson-inference/tools
./download-models.sh
```

You can select the models you want, or run the tool again later to download more models another time.

For now, the default models is enough.

![download-models](/assets/img/for_posts/2023-04-05/download-models.jpg)


# Install Pytorch (optional)

This step is optional, and if you don't wish to do the transfer learning steps on Jetson Nano, you don't need to install PyTorch and can skip this step.

```bash
cd ../build
./install-pytorch.sh
```

Select the PyTorch package versions for Python 2.7 and/or Python 3.6 that you want installed and hit Enter to continue.

![pytorch-installer](/assets/img/for_posts/2023-04-05/pytorch-installer.jpg)

You can also run this tool again later if you decide that you want to install PyTorch at another time.

# Install torchvision (optional, not completed yet)

## Why build torchvision from source

In the last step, torchvision will be installed automatedly. However, in my case, run `import torchvision` will raise error.

According to the official reply in [this post on Nvidia developer forum](https://forums.developer.nvidia.com/t/installing-torchvision/245286):

> TorchVision needs to build from the source on Jetson. You can find the steps and corresponding version below: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 > Installation > Torchvision

## Official instructions

So open [the post mentioned above](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048), and select Instructions > Installation > Torchvision in the top floor, get these procedures:

```bash
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch <version> https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
cd torchvision
export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version  
python3 setup.py install --user
cd ../  # attempting to load torchvision from build dir will result in import error
pip install 'pillow<7' # always needed for Python 2.7, not needed torchvision v0.5.0+ with Python 3.6
```

## DEBUG: Error: 'libavformat/avformat.h: No such file or directory.'

However, `python3 setup.py install --user` may raise an error:

```
Error: 'libavformat/avformat.h: No such file or directory.'
```

Some posts give this solution: `sudo apt-get install libavformat53 libavformat-dev libavcodec53`, which is worth a try.

While for me, `ffmpeg` have to be reinstalled from source, so that some libraries related like libavformat will be correctly set up in my system.

To do so, just clone ffmpeg source and checkout to tag `n4.4` in your home path:

```bash
mkdir ./ffmpeg_sources
cd ffmpeg_sources
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/
git checkout tags/n4.4
```

> Why tag `n4.4`? Refer to [this post](https://forum.opencv.org/t/error-avstream-aka-struct-avstream-has-no-member-named-codec/3506/11).

> The up-to-date version of ffmpeg would cause `error:'const AVSubtitleRect {aka const}' has no menber named 'pict'` in torchvision setup step.

And then follow the steps in [ffmpeg official installation instruction](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu), except the `./configure` step, which should be modified to:

```bash
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure \
  --enable-shared
  --disable-static
  --enable-nonfree
  --prefix="$HOME/ffmpeg_build" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I$HOME/ffmpeg_build/include" \
  --extra-ldflags="-L$HOME/ffmpeg_build/lib" \
  --extra-libs="-lpthread -lm" \
  --ld="g++" \
  --bindir="$HOME/bin" \
  --enable-gpl \
  --enable-gnutls \
  --disable-libaom \
  --enable-libass \
  --enable-libfdk-aac \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --disable-libsvtav1 \
  --disable-libdav1d \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-nonfree
```

> Why `enable-shared`? Refer to [this post](https://forum.opencv.org/t/error-avstream-aka-struct-avstream-has-no-member-named-codec/3506/11) again. 
> 
> btw, I disable `libaom`, `libsvtav1` and `libdav1d` as these libs are not necessary yet hard to install.

After the ffmpeg installation steps, there would be three directories in your home path:

* ffmpeg_sources, which contains sources files.
* ffmpeg_build, which contains lib files.
* bin, which contains executable files.

According the code in `torchvision setup.py file`, move the whole `bin` directory to `ffmpeg_build/`, and add `ffmpeg_build` to PATH, so that the setup file could find the ffmpeg executable and ffmpeg-related libs:

```bash
export PATH=$PATH:/home/xxx/ffmpeg_build/bin
export PALD_LIBRARY_PATHTH=$LD_LIBRARY_PATH:/home/xxx/ffmpeg_build/lib
```

Your can save this line in your `~/.bashrc` file so that the change will be valid upon login.

## DEBUG： subprocess.CalledProcessError: Command ‘[‘ninja’, ‘-v’]’ returned non-zero exit status 1

Run `python3 setup.py install --user` again, another error come out:

```
subprocess.CalledProcessError: Command ‘[‘ninja’, ‘-v’]’ returned non-zero exit status 1
```

According to some posts and docs, like [issue 37707 of pytorch](https://github.com/pytorch/pytorch/issues/37707):

> The new default setting of trying to use ninja to build extensions faster breaks existing C++/CUDA extensions. The reason appears to be that it does not forward the include directories specified to CppExtension or CUDAExtension classes to the ninja compiler properly.

So we can just use `cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}` in the setup function to fix tis error. 

> Some other posts suggest that change `[‘ninja’, ‘-v’]` to `[‘ninja’, ‘--v’]` may help. but in my case this operate would raise another error.


# Debug for CMake error in Jetson Inference compiling

Running cmake command directly would lead to this error:

```
CMake Error at python/bindings_python_3.6/cmake_install.cmake:89 (file):
  file INSTALL cannot find
  "/home/nvidia/jetson-inference/python/bindings/../python/jetson".
```

And this bug can be solved according to [issue 460 of Jetson Inference](https://github.com/dusty-nv/jetson-inference/issues/460)
.

First, rename：

* `jetson-inference/python/python/Jetson` to `jetson-inference/python/python/jetson`

* `jetson-inference/utils/python/python/Jetson` to `jetson-inference/python/python/jetson` 

Just lowcase the first 
letter of Jetson.

Then, comment out:

*`jetson-inference/python/bindings/CmakeList.txt` line 56, 57

*`jetson-inference/utils/python/bindings//CmakeList.txt` line 76, 77

whose content is:

```cmake
 56 install(DIRECTORY ../python/Jetson DESTINATION ${PYTHON_BINDING_INSTALL_DIR})
 57 install(DIRECTORY ../python/Jetson DESTINATION ${PYTHON_BINDING_INSTALL_DIR}) 
```

# Compiling the Project

Now your can go on compiling. Ensuring that you are in `jetson-inference/build/` directory, and run:

```
$ cmake ../
$ make -j4
$ sudo make install
$ sudo ldconfig
```

And congradulations! You have successfully install Jetson Inference on your Jetson Nano.

Now you can run some examples according to [this doc](https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md#detecting-objects-from-the-command-line), and here are my test output:

![123](/assets/img/for_posts/2023-04-05/pedestrians.gif)

