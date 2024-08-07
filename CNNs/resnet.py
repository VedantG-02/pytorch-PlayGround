import torch
import torch.nn as nn
from torchsummary import summary


# input image channels : 3
# input image dims : 224 X 224
class ResNet(nn.Module):
    def __init__(self, arch:str, in_c=3, n_classes=1000):
        '''
            Args:
            - arch (str) : specific architecture of resnet
            - in_c (int) : number of input image channels
            - n_classes (int) : number of output classes
        '''
        super(ResNet, self).__init__()
        # variants of resnet in terms of number of residual blocks; see paper for more details
        architectures = {
            'resnet-50' : [3, 4, 6, 3],
            'resnet-101' : [3, 4, 23, 3],
            'resnet-152' : [3, 8, 36, 3]
        }   
        self.in_c = in_c
        self.n_classes = n_classes

        self.conv1 = convBlock(in_c=in_c, out_c=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        res_in_c = 64
        res_n_channels = 64
        layers = []
        for id, num_blocks in enumerate(architectures[arch]):
            for block in range(num_blocks):
                layers.append(residualBlock(
                    in_c=res_in_c if block==0 else 4*res_n_channels, n_channels=res_n_channels, ds=(block==0 and (not id==0))
                ))
            res_in_c = 4*res_n_channels
            res_n_channels = 2*res_n_channels        
        self.conv_all = nn.Sequential(*layers)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(res_n_channels*2, n_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv_all(x)
        x = self.linear(self.avgpool(x))
        return x

class residualBlock(nn.Module):
    def __init__(self, in_c, n_channels, ds=1):
        '''
            Using identity shortcuts only and not projection shortcuts (TODO)
        
            Args:
            - in_c (int) : number of input channels to residual block
            - n_channels (int) : internal conv channels
            - ds (bool/int) : flag to downsample input
        '''
        super(residualBlock, self).__init__()
        self.in_c = in_c
        self.n_channels = n_channels
        self.ds = ds
        self.act = nn.ReLU()

        layers = []
        # res block for ResNet-50, ResNet-101 and ResNet-152 
        layers.append(convBlock(in_c=in_c, out_c=n_channels, kernel_size=1, stride=1+ds, padding=0))
        layers.append(convBlock(in_c=n_channels, out_c=n_channels, kernel_size=3, stride=1, padding=1))
        layers.append(convBlock(in_c=n_channels, out_c=n_channels*4, kernel_size=1, stride=1, padding=0, use_act=False))
        
        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        out = self.residual(x)
        # print(f"output shape: {out.shape}, input shape: {x.shape}") # uncomment both to check if shortcuts are correctly getting added 
        if self.ds == 0 and self.in_c == self.n_channels*4:
            # print(f"    - identity shortcut added\n")
            out = out + x
        out = self.act(out) # apply relu
        return out

class convBlock(nn.Module):
    def __init__(self, in_c, out_c, use_act=True, **kwargs):
        '''
            Args:
            - in_c (int) : number of input channels to conv layer
            - out_c (int) : number of output channels
            - use_act (bool) : apply activation or not
        '''
        super(convBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_c, out_c, **kwargs)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x))) if self.use_act else self.bn(self.conv(x))
        return x


def test():
    arch = 'resnet-50' # change 
    bs = 4
    n_classes = 1000
    in_c = 3
    x = torch.rand((bs, in_c, 224, 224)).cuda()
    model = ResNet(arch=arch, in_c=in_c, n_classes=n_classes).cuda()

    out = model(x)

    assert out.shape == (bs, n_classes), f"{arch} output shape doesn't match"

    summary(model, (in_c, 224, 224))
    
    print("\n# ---Testing Done--- #\n")


if __name__ == '__main__':
    test()