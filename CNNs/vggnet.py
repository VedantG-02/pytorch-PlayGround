import torch
import torch.nn as nn
from torchsummary import summary


# input image channels : 3
# input image dims : 224 X 224
class VGGNet(nn.Module):
    def __init__(self,  arch:str, in_c=3, n_classes=1000):
        '''
            Args:
            - arch (str) : specific architecture of vgg model; either of vgg11, vgg13, vgg16 or vgg19
            - in_c (int) : number of input image channels
            - n_classes (int) : number of output classes
        '''
        super(VGGNet, self).__init__()
        # variants of vgg models with 3X3 conv kernel
        architectures = {
            'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        }
        self.in_c = in_c
        self.n_classes = n_classes

        self.conv = self.convLayers(architectures[arch])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes)
        )

    def convLayers(self, architecture):
        layers = []
        in_c = self.in_c
        for layer in architecture:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_c = layer
                layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU())
                in_c = out_c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(self.conv(x))
        return x


def test():
    arch = 'VGG19' # change 
    bs = 16
    n_classes = 1000
    in_c = 3
    x = torch.rand((bs, in_c, 224, 224)).cuda()
    model = VGGNet(arch=arch, in_c=in_c, n_classes=n_classes).cuda()
    
    out = model(x)

    assert out.shape == (bs, n_classes), f"{arch}Net output shape doesn't match"

    summary(model, (in_c, 224, 224))

    print("\n# ---Testing Done--- #\n")


if __name__ == '__main__':
    test()