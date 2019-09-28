"""
Copyright 2017 the shiftnet_cuda_v2 authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import model.cifar10.shiftnet_cuda_v2.shiftnet_cuda as shiftnet_cuda
import torch
from torch.autograd import Function, Variable


class ShiftFn(Function):
  @staticmethod
  def forward(ctx, src):
    dst = torch.cuda.FloatTensor(src.size())
    ret = shiftnet_cuda.moduloshift3x3_nchw(src, dst)
    assert ret == 1
    return dst

  @staticmethod
  def backward(ctx, grad_dst):
    grad_src = torch.cuda.FloatTensor(grad_dst.data.size())
    ret = shiftnet_cuda.moduloshift3x3bwd_nchw(grad_dst.data, grad_src)
    assert ret == 1
    return Variable(grad_src, requires_grad=grad_dst.requires_grad)

class GenericShiftFn(Function):
  @staticmethod
  def forward(ctx, src, kernel_size, dilate_factor):
    ctx.kernel_size = kernel_size
    ctx.dilate_factor = dilate_factor
    dst = torch.cuda.FloatTensor(src.size())
    ret = shiftnet_cuda.moduloshiftgeneric_nchw(src, dst, kernel_size, dilate_factor, 1)
    assert ret == 1, "GenericShiftFn: forward: invalid args, your kernel or dilation are probably too large"
    return dst

  @staticmethod
  def backward(ctx, grad_dst):
    grad_src = torch.cuda.FloatTensor(grad_dst.data.size())
    ret = shiftnet_cuda.moduloshiftgeneric_nchw(grad_dst.data, grad_src, ctx.kernel_size, ctx.dilate_factor, -1)
    assert ret == 1, "GenericShiftFn: backward: invalid args, your kernel or dilation are probably too large"
    return Variable(grad_src, requires_grad=grad_dst.requires_grad), None, None

class Shift3x3_cuda(torch.nn.Module):
    def __init__(self):
        super(Shift3x3_cuda, self).__init__()

    def forward(self, x):
        return ShiftFn.apply(x)

class GenericShift_cuda(torch.nn.Module):
    def __init__(self, kernel_size, dilate_factor=1):
        super(GenericShift_cuda, self).__init__()
        self._kernel_size = kernel_size
        self._dilate_factor = dilate_factor

    def forward(self, x):
        return GenericShiftFn.apply(x, self._kernel_size, self._dilate_factor)
