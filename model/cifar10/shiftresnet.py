
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cifar10.shiftnet_cuda_v2.nn import GenericShift_cuda


class DWBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, expansion=1):
        super(DWBlock, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = mid_planes = int(out_planes * self.expansion)

        assert self.expansion == 1

        self.dw1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, bias=False, padding=1, groups=in_planes)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.dw2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, bias=False, padding=1, groups=mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                  in_planes, out_planes, kernel_size=1, stride=stride,
                  bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.in_planes * 9 + \
                int_h * int_w * self.in_planes * self.mid_planes + \
                out_h * out_w * self.mid_planes * 9 + \
                out_h * out_w * self.mid_planes * self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes * self.out_planes * out_h * out_w
        return flops

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(self.dw1(x))))
        self.int_nchw = x.size()
        x = F.relu(self.bn2(self.conv2(self.dw2(x))))
        self.out_nchw = x.size()
        x += shortcut
        return x

class GroupBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, groups=1, expansion=1):
        super(GroupBlock, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = mid_planes = int(out_planes * self.expansion)
        self.groups=groups

        assert self.expansion == 1

        self.group1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, bias=False, padding=1, groups=groups)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.group2=nn.Conv2d(mid_planes, mid_planes, kernel_size=3, bias=False, stride=1, padding=1, groups=groups)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                  in_planes, out_planes, kernel_size=1, stride=stride,
                  bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.in_planes * self.in_planes * 9 / self.groups + \
                int_h * int_w * self.in_planes * self.mid_planes + \
                out_h * out_w * self.mid_planes * self.mid_planes * 9 / self.groups + \
                out_h * out_w * self.mid_planes * self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes * self.out_planes * out_h * out_w
        return flops

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(self.group1(x))))
        self.int_nchw = x.size()
        x = F.relu(self.bn2(self.conv2(self.group2(x))))
        self.out_nchw = x.size()
        x += shortcut
        return x

class ShiftBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, expansion=1):
        super(ShiftBlock, self).__init__()
        self.expansion = expansion
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = mid_planes = int(out_planes * self.expansion)

        self.shift1 = GenericShift_cuda(kernel_size=3, dilate_factor=1)
        self.conv1 = nn.Conv2d(
            in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.shift2 = GenericShift_cuda(kernel_size=3, dilate_factor=1)
        self.conv2 = nn.Conv2d(
            mid_planes, out_planes, kernel_size=1, bias=False, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                  in_planes, out_planes, kernel_size=1, stride=stride,
                  bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (_, _, out_h, out_w) = self.int_nchw, self.out_nchw
        flops = int_h * int_w * self.in_planes * self.mid_planes + \
                out_h * out_w * self.mid_planes * self.out_planes
        if len(self.shortcut) > 0:
            flops += self.in_planes * self.out_planes * out_h * out_w
        return flops

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(self.shift1(x))))
        self.int_nchw = x.size()
        x = F.relu(self.bn2(self.conv2(self.shift2(x))))
        self.out_nchw = x.size()
        x += shortcut
        return x


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, reduction=1, num_classes=10):
        super(ResNet, self).__init__()
        self.reduction = float(reduction) ** 0.5
        self.num_classes = num_classes
        self.in_planes = int(16 / self.reduction)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(32 / self.reduction), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(64 / self.reduction), num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(64 / self.reduction), num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        planes = int(planes)
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def flops(self):
        if not hasattr(self, 'int_nchw'):
            raise UserWarning('Must run forward at least once')
        (_, _, int_h, int_w), (out_h, out_w) = self.int_nchw, self.out_hw
        flops = 0
        for mod in (self.layer1, self.layer2, self.layer3):
            for layer in mod:
                flops += layer.flops()
        return int_h*int_w*3*3*16*3 + out_w*self.num_classes + flops

    def forward(self, x, is_pool=True):
        out = F.relu(self.bn1(self.conv1(x)))
        self.int_nchw = out.size()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        if is_pool:
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            self.out_hw = out.size()
            out = self.fc(out)

        return out

class ResNetEnsemble_concat_4(nn.Module):
    def __init__(self, block, structure, is_activate=True, num_classes=10):
        super(ResNetEnsemble_concat_4, self).__init__()
        self.inplanes = 16

        self.resnet_m1 = ResNet(block, structure, num_classes=num_classes)
        self.resnet_m2 = ResNet(block, structure, num_classes=num_classes)
        self.resnet_m3 = ResNet(block, structure, num_classes=num_classes)
        self.resnet_m4 = ResNet(block, structure, num_classes=num_classes)

        self.is_activate = is_activate

        if self.is_activate:
            self.norm = nn.BatchNorm2d(256)
            self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        features_m1 = self.resnet_m1(x, is_pool=False)
        features_m2 = self.resnet_m2(x, is_pool=False)
        features_m3 = self.resnet_m3(x, is_pool=False)
        features_m4 = self.resnet_m4(x, is_pool=False)
        features_e = torch.cat((features_m1, features_m2, features_m3, features_m4), 1)

        logits_m1 = self.resnet_m1.avgpool(features_m1)
        logits_m1 = logits_m1.view(logits_m1.size(0), -1)
        logits_m1 = self.resnet_m1.fc(logits_m1)

        logits_m2 = self.resnet_m2.avgpool(features_m2)
        logits_m2 = logits_m2.view(logits_m2.size(0), -1)
        logits_m2 = self.resnet_m2.fc(logits_m2)

        logits_m3 = self.resnet_m3.avgpool(features_m3)
        logits_m3 = logits_m3.view(logits_m3.size(0), -1)
        logits_m3 = self.resnet_m3.fc(logits_m3)

        logits_m4 = self.resnet_m4.avgpool(features_m4)
        logits_m4 = logits_m4.view(logits_m4.size(0), -1)
        logits_m4 = self.resnet_m4.fc(logits_m4)

        if self.is_activate:
            logits_e = self.relu(self.norm(features_e))
        else:
            logits_e = features_e

        logits_e = self.avgpool(logits_e)
        logits_e = logits_e.view(logits_e.size(0), -1)
        logits_e = self.fc(logits_e)

        return (logits_m1, logits_m2, logits_m3, logits_m4), logits_e




def ShiftResNet56_od(expansion=1, num_classes=10, num_stu=2, groups=1):
    block = lambda in_planes, out_planes, stride: \
        ShiftBlock(in_planes, out_planes, stride, expansion=expansion)

    if num_stu==4:
        return ResNetEnsemble_concat_4(block, [9, 9, 9], num_classes=num_classes)

def GroupResNet56_od(groups=1, expansion=1, num_classes=10, num_stu=2):
    block = lambda in_planes, out_planes, stride: \
        GroupBlock(in_planes, out_planes, stride, groups=groups, expansion=expansion)

    if num_stu == 4:
        return ResNetEnsemble_concat_4(block, [9, 9, 9], num_classes=num_classes)

def DWResNet56_od(expansion=1, num_classes=10, num_stu=2, groups=1):
    block = lambda in_planes, out_planes, stride: \
        DWBlock(in_planes, out_planes, stride, expansion=expansion)

    if num_stu == 4:
        return ResNetEnsemble_concat_4(block, [9, 9, 9], num_classes=num_classes)


