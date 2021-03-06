import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet_original(nn.Module):
    """
    Original AlexNet Architecture for ImageNet.
    * No Responce Normalization
    """
    def __init__(self, num_classes=1000):
        super(AlexNet_original, self).__init__()
        self.features = nn.Sequential(
            # layer 1
            nn.Conv2d(3, 96, 11, 4),              # [227, 227, 3] -> [55, 55, 96]
            nn.MaxPool2d(3, 2),                   # [55, 55, 96] -> [27, 27, 96]
            nn.ReLU(True),
            # layer 2
            nn.Conv2d(96, 256, 3, 1, padding=1),  # [27, 27, 96] -> [27, 27, 256]
            nn.MaxPool2d(3, 2),                   # [27, 27, 256] -> [13, 13, 256]
            nn.ReLU(True),
            # layer 3
            nn.Conv2d(256, 384, 3, 1, padding=1), # [13, 13, 128] -> [13, 13, 384]
            nn.ReLU(True),
            # layer 4
            nn.Conv2d(384, 384, 3, 1, padding=1), # [13, 13, 384] -> [13, 13, 384]
            nn.ReLU(True),
            # layer 5
            nn.Conv2d(384, 256, 3, 1, padding=1), # [13, 13, 384] -> [13, 13, 256]
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            # layer 6
            nn.Dropout(0.5),
            nn.Linear(13*13*256, 4096),
            nn.ReLU(True),
            # layer 7
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # layer 8
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    """
    AlexNet adapted for Cifar-10.
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, padding=2),  # 32->32
            nn.MaxPool2d(3, 2),                 # 32->15
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, padding=2), 
            nn.MaxPool2d(3, 2),                 # 15->7
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(7*7*32, 10),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

class AlexNet2(nn.Module):
    """
    AlexNet adapted for Cifar-10.
    """
    def __init__(self, num_classes=10):
        super(AlexNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, padding=2),  # 32->32
            nn.MaxPool2d(3, 2),                 # 32->15
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, 1, padding=2), 
            nn.MaxPool2d(3, 2),                 # 15->7
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, padding=1),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(7*7*32, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x

## ResNet

class ResBlock_original(nn.Module):
    """
    ResNet Basic Block
    [B, Fin, H, W] -> [B, Fout, H, W]
    """
    expansion = 1
    def __init__(self, Fin, Fout, stride):
        super(ResBlock_original, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(Fin, Fout, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(Fout),
            nn.ReLU(True),
            nn.Conv2d(Fout, Fout, 3, padding=1, bias=False),
            nn.BatchNorm2d(Fout),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(Fin, Fout, 1, stride, bias=False),
            nn.BatchNorm2d(Fout),
        ) if Fin != Fout or stride != 1 else lambda x: x

    def forward(self, x):
        res = self.residual(x)
        x = self.feature(x)
        x = x + res
        x = F.relu(x, True)
        return x

class BottleNeck_original(nn.Module):
    """
    ResNet BottleNeck Block
    [B, Fin, H, W] -> [B, Fout*expansion, H, W]
    """
    expansion = 4 
    def __init__(self, Fin, Fout, stride):
        super(BottleNeck_original, self).__init__()
        self.feature = nn.Sequential(
            # layer1 eg. 256 -> 64
            nn.Conv2d(Fin, Fout, 1, bias=False),
            nn.BatchNorm2d(Fout),
            nn.ReLU(True),
            # layer2 eg. 64 -> 64
            nn.Conv2d(Fout, Fout, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(Fout),
            nn.ReLU(True),
            # layer3 eg. 64 -> 256
            nn.Conv2d(Fout, self.expansion*Fout, 1, bias=False),
            nn.BatchNorm2d(self.expansion*Fout),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(Fin, self.expansion*Fout, 1, stride, bias=False),
            nn.BatchNorm2d(self.expansion*Fout),
        ) if Fin != self.expansion*Fout or stride != 1 else lambda x: x

    def forward(self, x):
        res = self.residual(x)
        x = self.feature(x)
        x = x + res
        x = F.relu(x, True)
        return x

class ResBlock_preActivation(nn.Module):
    """
    ResNet Basic Block
    [B, Fin, H, W] -> [B, Fout, H, W]
    """
    expansion = 1
    def __init__(self, Fin, Fout, stride):
        super(ResBlock_preActivation, self).__init__()
        self.feature = nn.Sequential(
            nn.BatchNorm2d(Fin),
            nn.ReLU(True),
            nn.Conv2d(Fin, Fout, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(Fout),
            nn.ReLU(True),
            nn.Conv2d(Fout, Fout, 3, padding=1, bias=False),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(Fin, Fout, 1, stride, bias=False),
            nn.BatchNorm2d(Fout),
        ) if Fin != Fout or stride != 1 else lambda x: x

    def forward(self, x):
        res = self.residual(x)
        x = self.feature(x)
        x = x + res
        return x

class BottleNeck_preActivation(nn.Module):
    """
    ResNet BottleNeck Block
    [B, Fin, H, W] -> [B, Fout*expansion, H, W]
    """
    expansion = 4 
    def __init__(self, Fin, Fout, stride):
        super(BottleNeck_preActivation, self).__init__()
        self.feature = nn.Sequential(
            # layer1 eg. 256 -> 64
            nn.BatchNorm2d(Fin),
            nn.ReLU(True),
            nn.Conv2d(Fin, Fout, 1, bias=False),
            # layer2 eg. 64 -> 64
            nn.BatchNorm2d(Fout),
            nn.ReLU(True),
            nn.Conv2d(Fout, Fout, 3, stride, padding=1, bias=False),
            # layer3 eg. 64 -> 256
            nn.BatchNorm2d(Fout),
            nn.ReLU(True),
            nn.Conv2d(Fout, self.expansion*Fout, 1, bias=False),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(Fin, self.expansion*Fout, 1, stride, bias=False),
            nn.BatchNorm2d(self.expansion*Fout),
        ) if Fin != self.expansion*Fout or stride != 1 else lambda x: x

    def forward(self, x):
        res = self.residual(x)
        x = self.feature(x)
        x = x + res
        return x

def ResNet18():
    return ResNet(ResBlock_original, [2,2,2,2])

def ResNet18_pre():
    return ResNet(ResBlock_preActivation, [2,2,2,2])

def ResNet56():
    return ResNet(BottleNeck_original, [3,4,6,3])

def ResNet56_pre():
    return ResNet(BottleNeck_preActivation, [3,4,6,3])

class ResNet(nn.Module):
    """
    ResNet Architecture.
    Usage: ResNet(BlockType, [a, b, c, d])
           eg. ResNet18: ResNet(BasicBlock, [2,2,2,2]) 18=1+(2+2+2+2)*2+1
           eg. ResNet34: ResNet(BasicBlock, [3,4,6,3])
           eg. ResNet56: ResNet(BottleNeck, [3,4,6,3])
           eg. ResNet101: ResNet(BottleNeck, [3,4,23,3])
           eg. ResNet152: ResNet(BottleNeck, [3,8,36,3])
           
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.Fin = 64

        self.layer0 = nn.Sequential(                                        # [32,32,3] -> [32,32,64]
            nn.Conv2d(3, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)   # [32,32,64] -> [16,16,64]
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)  # [16,16,64] -> [16,16,128]
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)  # [16,16,128] -> [8,8,256]
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)  # [8,8,256] -> [4,4,512]
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, Fout, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.Fin, Fout, stride))
            self.Fin = Fout * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x



## ResNext

class BottleNeck_ResNeXt(nn.Module):
    """
    ResNeXt BottleNeck Block
    [B, Fin, H, W] -> [B, expansion*Fout, H, W]
    """
    expansion = 2
    def __init__(self, Fin, Fout, stride=1, group=1):
        super(BottleNeck_ResNeXt, self).__init__()
        assert Fout % group == 0
        width = Fout // group

        self.feature = nn.Sequential(
            nn.Conv2d(Fin, Fout, 1, bias=False),
            nn.BatchNorm2d(Fout),
            nn.ReLU(True),
            nn.Conv2d(Fout, Fout, 3, stride, padding=1, bias=False, groups=group),
            nn.BatchNorm2d(Fout),
            nn.ReLU(True),
            nn.Conv2d(Fout, self.expansion*Fout, 1, bias=False),
            nn.BatchNorm2d(self.expansion*Fout),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(Fin, self.expansion*Fout, 1, stride, bias=False),
            nn.BatchNorm2d(self.expansion*Fout),
        ) if Fin != self.expansion*Fout or stride != 1 else lambda x: x

    def forward(self, x):
        res = self.residual(x)
        x = self.feature(x)
        x = x + res
        x = F.relu(x, True)
        return x

def ResNeXt2x():
    return ResNeXt(2, [3,4,6,3])

class ResNeXt(nn.Module):
    """
    ResNeXt Architecture.
    cardinality is called group here.
    Usage: ResNeXt(group, [a, b, c, d])
    """
    def __init__(self, group, num_blocks, block=BottleNeck_ResNeXt, num_classes=10):
        super(ResNeXt, self).__init__()
        self.Fin = 64

        self.layer0 = nn.Sequential(                                        # [32,32,3] -> [32,32,64]
            nn.Conv2d(3, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1, group=group)   # [32,32,64] -> [16,16,64]
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2, group=group)  # [16,16,64] -> [16,16,128]
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2, group=group)  # [16,16,128] -> [8,8,256]
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2, group=group)  # [8,8,256] -> [4,4,512]
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, Fout, num_blocks, stride, group):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.Fin, Fout, stride=stride, group=group))
            self.Fin = Fout * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

## DenseNet

class BottleNeck_DenseNet(nn.Module):
    """
    DenseNet BottleNeck Block
    [B, Fin, H, W] -> [B, Fin+Fout, H, W]
    """
    expansion = 4
    def __init__(self, Fin, Fout):
        super(BottleNeck_DenseNet, self).__init__()
        self.Fin = Fin
        self.Fout = Fout
        self.feature = nn.Sequential(
            nn.BatchNorm2d(Fin),
            nn.ReLU(),
            nn.Conv2d(Fin, self.expansion*Fout, 1, bias=False),
            nn.BatchNorm2d(self.expansion*Fout),
            nn.ReLU(),
            nn.Conv2d(self.expansion*Fout, Fout, 3, padding=1, bias=False),
        )

    def forward(self, x):
        res = x
        x = self.feature(x)
        x = torch.cat([x, res], dim=1)
        return x

class Transition_DenseNet(nn.Module):
    def __init__(self, Fin, reduction):
        super(Transition_DenseNet, self).__init__()
        self.Fin = Fin
        self.Fout = int(math.floor(Fin * reduction))
        self.feature = nn.Sequential(
            nn.BatchNorm2d(Fin),
            nn.ReLU(),
            nn.Conv2d(Fin, self.Fout, 1, bias=False),
            nn.AvgPool2d(2),
        )
    
    def forward(self, x):
        return self.feature(x)

def DenseNet121():
    return DenseNet(BottleNeck_DenseNet, [6,12,24,16], 12, 0.5)

class DenseNet(nn.Module):
    def __init__(self, block, num_blocks, growth_rate, reduction, num_classes=10):
        super(DenseNet, self).__init__()
        self.block = block
        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.reduction = reduction

        planes = 2 * growth_rate
        self.layer0 = nn.Conv2d(3, planes, 3, padding=1, bias=False)

        self.dense0 = self.make_layer(planes, num_blocks[0])
        planes += num_blocks[0] * growth_rate
        self.trans0 = Transition_DenseNet(planes, reduction)
        planes = self.trans0.Fout

        self.dense1 = self.make_layer(planes, num_blocks[1])
        planes += num_blocks[1] * growth_rate
        self.trans1 = Transition_DenseNet(planes, reduction)
        planes = self.trans1.Fout

        self.dense2 = self.make_layer(planes, num_blocks[2])
        planes += num_blocks[2] * growth_rate
        self.trans2 = Transition_DenseNet(planes, reduction)
        planes = self.trans2.Fout

        self.dense3 = self.make_layer(planes, num_blocks[3])
        planes += num_blocks[3] * growth_rate
        
        self.pool = nn.Sequential(
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.AvgPool2d(4),
        )
        
        self.fc = nn.Linear(planes, num_classes)

    
    def make_layer(self, Fin, num_block):
        layers = []
        for i in range(num_block):
            layers.append(self.block(Fin, self.growth_rate))
            Fin += self.growth_rate
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.layer0(x)
        x = self.trans0(self.dense0(x))
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.fc(x)
        return x