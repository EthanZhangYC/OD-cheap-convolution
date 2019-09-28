# Training convolutional neural networks with cheap convolutions and online distillation

PyTorch implementation for OD-cheap-convolution.



![OD-framework](https://user-images.githubusercontent.com/47294246/54805147-021eb500-4cb1-11e9-85ac-861ecbada3e1.png)

Illustration of our online distillation method with cheap convolutions. (a) The original network with standard convolutions. (b) Online distillation is performed on the compressed network with cheap convolutions to further improve its performance. First, a student network is formed by replacing its standard convolution with the cheap ones. Then, a strong teacher network is constructed online by concatenating the output features Z(i) from the same multiple student networks and adding the new classifier. During the training, mutual learning is conducted to improve the performance between teacher and students. For testing, the best student model in validation set is selected to be a compressed model. (Best viewed in color.)



## Abstract

The large memory and computation consumption in convolutional neural networks (CNNs) has been one of the main barriers for deploying them on resource-limited systems. To this end, most cheap convolutions (e.g., group convolution, depth-wise convolution and shift convolution) have been recently used for memory and computation reduction but with the specific architecture designing. Furthermore, it results in a low discriminability of the compressed networks by directly replacing the standard convolution with these cheap ones. In this paper, we propose to use knowledge distillation to improve the performance of the compact student networks with the cheap convolutions. In our case the teacher is a network with the standard convolution, while the student is a simple transformation of the teacher architecture without complicated redesigning. In particular, we propose a novel online distillation method, which online constructs the teacher network without pre-training and conducts mutual learning between the teacher and student network, to improve the performance of student model. Extensive experiments demonstrate that the proposed approach achieves superior performance to simul- taneously reduce memory and computation overhead of cutting-edge CNNs on different datasets, including CIFAR-10/100 and ImageNet ILSVRC 2012, compared to the state-of-the-art CNN compression and acceleration methods.


## Running Code

In this code, you can run our models on CIFAR10 dataset. The code has been tested by Python 3.6, [Pytorch 0.4.1](https://pytorch.org/) and CUDA 9.0 on Ubuntu 16.04.



### Run examples

The scripts of training and fine-tuning are provided  in the `run.sh`, please kindly uncomment the appropriate line in `run.sh` to execute the training and fine-tuning.

```shell
sh run.sh
```



**For training**, change the `teacher_dir` to the place where the pretrained model is located. 

```shell
# ResNet-56
MIU=1
LAMBDA=0.8
python main.py \
--teacher_dir [pre-trained model dir] \
--arch resnet --teacher_model resnet_56 --student_model resnet_56_sparse \
--lambda $LAMBDA --miu $MIU \
--job_dir 'experiment/resnet/lambda_'$LAMBDA'_miu_'$MIU
```

After training, checkpoints and loggers can be found in the `job_dir`, The pruned model of best performance will be named `[arch]_pruned_[pruned_num].pt`. For example: `resnet_pruned_11.pt`



**For fine-tuning**, change the `refine` to the place where the pruned model is allowed to be fine-tuned. 

```shell
# ResNet-56
python finetune.py \
--arch resnet --lr 1e-4 \
--refine experiment/resnet/lambda_0.8_miu_1/resnet_pruned_11.pt \
--job_dir experiment/resnet/ft_lambda_0.8_miu_1/ \
--pruned 
```



## Tips

If you find any problems, please feel free to contact to the authors (jiaoxie1990@126.com (Jiao Xie), shaohuilin007@gmail.com(Shaohui Lin), ethan.zhangyc@gmail.com (Yichen Zhang), luolk@xmu.edu.cn (Linkai Luo)).
