# Training convolutional neural networks with cheap convolutions and online distillation

PyTorch implementation for OD-cheap-convolution.



![OD-framework](https://raw.githubusercontent.com/EthanZhangYC/OD-cheap-convolution/master/od_arch.png)

Illustration of our online distillation method with cheap convolutions. (a) The original network with standard convolutions. (b) Online distillation is performed on the compressed network with cheap convolutions to further improve its performance. First, a student network is formed by replacing its standard convolution with the cheap ones. Then, a strong teacher network is constructed online by concatenating the output features Z(i) from the same multiple student networks and adding the new classifier. During the training, mutual learning is conducted to improve the performance between teacher and students. For testing, the best student model in validation set is selected to be a compressed model. (Best viewed in color.)



## Abstract

The large memory and computation consumption in convolutional neural networks (CNNs) has been one of the main barriers for deploying them on resource-limited systems. To this end, most cheap convolutions (e.g., group convolution, depth-wise convolution and shift convolution) have been recently used for memory and computation reduction but with the specific architecture designing. Furthermore, it results in a low discriminability of the compressed networks by directly replacing the standard convolution with these cheap ones. In this paper, we propose to use knowledge distillation to improve the performance of the compact student networks with the cheap convolutions. In our case the teacher is a network with the standard convolution, while the student is a simple transformation of the teacher architecture without complicated redesigning. In particular, we propose a novel online distillation method, which online constructs the teacher network without pre-training and conducts mutual learning between the teacher and student network, to improve the performance of student model. Extensive experiments demonstrate that the proposed approach achieves superior performance to simul- taneously reduce memory and computation overhead of cutting-edge CNNs on different datasets, including CIFAR-10/100 and ImageNet ILSVRC 2012, compared to the state-of-the-art CNN compression and acceleration methods.


## Usage

In this code, you can run our code on CIFAR10 dataset. The code has been tested by Python 3.6, [Pytorch 0.4.1](https://pytorch.org/) and CUDA 9.0 on Ubuntu 16.04.


### Run examples

In our implementation, there are totally three kinds of cheap convolution, including shift operation, depthwise convolution and group convolution. You can feel free to select any one of it by the hyperparameter `block_type` for setting it to 'Shift', 'DW' or 'Group' correspondingly.

**For training**, change the `job_dir` to the path where you would like to save your checkpoint and logger files. 

```shell
python3 main_teacher.py \
--data_dir ./data \
--job_dir [saving_path] \
--block_type Shift \
--num_stu 4 \
--gpu 0
```

**For fine-tuning**, set the `resume` parameter to the path where you place the pretrain model or checkpoint. 

```shell
python3 main_teacher.py \
--data_dir ./data \
--job_dir [saving_path] \
--block_type DW \
--num_stu 4 \
--resume [checkpoint_path] \
--gpu 0
```

**For test-only**, you can also set the `test_only` to only validate the model without wasting time training. 

```shell
python3 main_teacher.py \
--data_dir ./data \
--job_dir [saving_path] \
--block_type DW \
--num_stu 4 \
--test_only \
--resume [checkpoint_path] \
--gpu 0
```

## Acknowledgement

This implementation largely borrows from blablabla...(unfinished)


