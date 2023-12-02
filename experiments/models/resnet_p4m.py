"""P4M equivariant ResNet architectures."""

import torch
import torch.nn as nn
from sepgroupy.gconv.g_splitgconv2d import gP4MConvP4M
from sepgroupy.gconv.gc_splitgconv2d import gcP4MConvP4M
from sepgroupy.gconv.splitgconv2d import P4MConvP4M, P4MConvZ2
from torchvision.models.resnet import ResNet


def conv3x3(layer, in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return layer(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return P4MConvP4M(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        base_width=64,
        conv_layer=None,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()

        if conv_layer is None:
            conv_layer = P4MConvP4M
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(conv_layer, inplanes, planes, stride)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(conv_layer, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity

        return x


class CustomResNet(ResNet):
    def __init__(
        self,
        block,
        layers,
        width,
        num_classes=1000,
        sep=None,
        zero_init_residual=False,
        width_per_group=64,
        norm_layer=None,
        groupcosetmaxpool=False,
    ):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        conv_layer = {None: P4MConvP4M, "g": gP4MConvP4M, "gc": gcP4MConvP4M}

        strides = [1, 2, 2, 2]
        channels = [width * 2**i for i in range(len(layers))]

        self.groupcosetmaxpool = groupcosetmaxpool

        self.inplanes = channels[0]
        self.base_width = width_per_group
        self.conv1 = P4MConvZ2(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(channels[-1])
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.layers.append(
                self._make_layer(
                    block, channels[i], layers[i], conv_layer[sep], stride=strides[i]
                )
            )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if groupcosetmaxpool:
            self.fc = nn.Linear(channels[-1] * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(channels[-1] * block.expansion * 8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the residual branch starts
        # with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, conv_layer, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.base_width,
                conv_layer,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    conv_layer=conv_layer,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)

        for layer in self.layers:
            x = layer(x)

        x = self.bn1(x)
        x = self.relu(x)
        x_fm = x.clone()

        n, nc, ns, nx, ny = x.shape

        if self.groupcosetmaxpool:
            x, _ = torch.max(x, dim=2)
        else:
            x = x.view(n, nc * ns, nx, ny)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, x_fm

    def forward(self, x):
        return self._forward_impl(x)


def P4MResNet18(width=23, **kwargs):
    return CustomResNet(BasicBlock, [2, 2, 2, 2], width, **kwargs)


def gcP4MResNet18(width=64, **kwargs):
    return CustomResNet(BasicBlock, [2, 2, 2, 2], width, sep="gc", **kwargs)


def P4MResNet44(width=12, **kwargs):
    return CustomResNet(BasicBlock, [7, 7, 7], width, **kwargs)


def gcP4MResNet44(width=35, **kwargs):
    return CustomResNet(BasicBlock, [7, 7, 7], width, sep="gc", **kwargs)


def getMaxCudaMem(model, in_shape):
    # Measure memory use
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    x = torch.rand(in_shape).cuda()
    torch.cuda.reset_peak_memory_stats()
    s = torch.cuda.max_memory_allocated()

    # Perform forward + backward pass
    y = m(x)
    label = torch.tensor([0, 0]).cuda()
    loss = criterion(y, label)
    loss.backward()

    mem = (torch.cuda.max_memory_allocated() - s) / 1024**2
    print("Max memory used: {:.2f} MB".format(mem))


if __name__ == "__main__":
    from torchinfo import summary

    m = P4MResNet18(num_classes=10)
    summary(m, (1, 3, 32, 32))

    m = gcP4MResNet18(num_classes=10)
    summary(m, (1, 3, 32, 32))

    m = P4MResNet44(num_classes=10)
    summary(m, (1, 3, 32, 32))

    m = gcP4MResNet44(num_classes=10)
    summary(m, (1, 3, 32, 32))
