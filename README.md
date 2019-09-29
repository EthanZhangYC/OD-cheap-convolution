# Training convolutional neural networks with cheap convolutions and online distillation

PyTorch implementation for OD-cheap-convolution.



![OD-framework](https://raw.githubusercontent.com/EthanZhangYC/OD-cheap-convolution/master/od_arch.png)

Illustration of our online distillation method with cheap convolutions.


## Usage

### Environment

In this code, you can run our code on CIFAR10 dataset. The code has been tested by Python 3.6, [Pytorch 0.4.1](https://pytorch.org/) and CUDA 9.0 on Ubuntu 16.04.

### Examples

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

The shift convolution is implemented by referring to [shiftresnet-cifar](https://github.com/alvinwan/shiftresnet-cifar).




