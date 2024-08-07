import torch
import torch.nn as nn
from torchsummary import summary


# input image channels : 3
# input image dims : 224 X 224
class GoogLeNet(nn.Module):
    def __init__(self, in_c=3, n_classes=1000):
        '''
            Args:
            - in_c (int) : number of input image channels
            - n_classes (int) : number of output classes
        '''
        super(GoogLeNet, self).__init__()
        self.conv1 = convBlock(in_c=in_c, out_c=64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = convBlock(in_c=64, out_c=64, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = convBlock(in_c=64, out_c=192, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3_a = inceptionBlock(in_c=192, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32)
        self.inception3_b = inceptionBlock(in_c=256, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4_a = inceptionBlock(in_c=480, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64)
        self.inception4_b = inceptionBlock(in_c=512, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64)
        self.inception4_c = inceptionBlock(in_c=512, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64)
        self.inception4_d = inceptionBlock(in_c=512, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64)
        self.inception4_e = inceptionBlock(in_c=528, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5_a = inceptionBlock(in_c=832, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)
        self.inception5_b = inceptionBlock(in_c=832, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.inception3_b(self.inception3_a(x)))
        x = self.inception4_a(x)
        x = self.inception4_b(x)
        x = self.inception4_c(x)
        x = self.inception4_d(x)
        x = self.inception4_e(x)
        x = self.pool4(x)
        x = self.avgpool(self.inception5_b(self.inception5_a(x)))
        x = self.linear(x)
        return x

class inceptionBlock(nn.Module):
    def __init__(self, in_c, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        '''
            Args:
            - in_c (int) : number of input image channels
            - out_1x1 (int) : number of output channels from path1
            - red_3x3 (int) : number of output channels from first conv block (1x1) in path2
            - out_3x3 (int) : number of output channels from second conv block (3x3) in path2
            - red_5x5 (int) : number of output channels from first conv block (1x1) in path3
            - out_5x5 (int) : number of output channels from second conv block (5x5) in path3
            - out_1x1pool (int) : number of output channels from path4  
        '''
        super(inceptionBlock, self).__init__()
        self.path1 = convBlock(in_c=in_c, out_c=out_1x1, kernel_size=1, stride=1, padding=0)
        self.path2 = nn.Sequential(
            convBlock(in_c=in_c, out_c=red_3x3, kernel_size=1, stride=1, padding=0),
            convBlock(in_c=red_3x3, out_c=out_3x3, kernel_size=3, stride=1, padding=1)
        )
        self.path3 = nn.Sequential(
            convBlock(in_c=in_c, out_c=red_5x5, kernel_size=1, stride=1, padding=0),
            convBlock(in_c=red_5x5, out_c=out_5x5, kernel_size=5, stride=1, padding=2)
        )
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            convBlock(in_c=in_c, out_c=out_1x1pool, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = torch.cat([self.path1(x), self.path2(x), self.path3(x), self.path4(x)], dim=1) # concat along channels dimension
        return x

class convBlock(nn.Module):
    def __init__(self, in_c, out_c, **kwargs):
        '''
            Args:
            - in_c (int) : number of input image channels
            - out_c (int) : number of output channels
        '''
        super(convBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, **kwargs)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv(x))
        return x


def test():
    bs = 16
    n_classes = 1000
    in_c = 3
    x = torch.rand((bs, in_c, 224, 224)).cuda()
    model = GoogLeNet(in_c=in_c, n_classes=n_classes).cuda()
    
    out = model(x)

    assert out.shape == (bs, n_classes), f"GoogLeNet output shape doesn't match"

    summary(model, (in_c, 224, 224))

    print("\n# ---Testing Done--- #\n")


if __name__ == '__main__':
    test()