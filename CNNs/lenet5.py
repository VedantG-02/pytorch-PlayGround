import torch 
import torch.nn as nn
from torchsummary import summary


# input image channels : 1
# input image dims : 32 X 32
class LeNet5(nn.Module):
    def __init__(self, in_c=1, n_classes=10):
        '''
            Args:
            - in_c (int) : number of input image channels
            - n_classes (int) : number of output classes
        '''
        super(LeNet5, self).__init__()
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_c, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = self.tanh(self.conv3(x))
        x = self.flatten(x)
        x = self.tanh(self.linear1(x))
        return self.linear2(x)


def test():
    bs = 16
    n_classes = 10
    in_c = 1
    x = torch.rand((bs, in_c, 32, 32)).cuda()
    model = LeNet5(in_c=in_c, n_classes=n_classes).cuda()
    
    out = model(x)

    assert out.shape == (bs, n_classes), "LeNet5 output shape doesn't match"

    summary(model, (in_c, 32, 32))

    print("\n# ---Testing Done--- #\n")


if __name__ == '__main__':
    test()