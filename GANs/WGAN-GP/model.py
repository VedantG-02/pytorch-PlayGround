import torch
import torch.nn as nn
from torchsummary import summary


# # generator and discriminator classes below; both FCNs
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_c=3):
        '''
            Args:
            - z_dim (int) : random vector (noise) dim
            - img_c (int) : number of image channels produced by gen
        '''
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            convBlock(in_c=z_dim, out_c=1024, disc=False, use_bn=True, use_act=True, kernel_size=4, stride=1, padding=0),
            convBlock(in_c=1024, out_c=512, disc=False, use_bn=True, use_act=True, kernel_size=4, stride=2, padding=1),
            convBlock(in_c=512, out_c=256, disc=False, use_bn=True, use_act=True, kernel_size=4, stride=2, padding=1),
            convBlock(in_c=256, out_c=128, disc=False, use_bn=True, use_act=True, kernel_size=4, stride=2, padding=1),
            convBlock(in_c=128, out_c=img_c, disc=False, use_bn=False, use_act=False, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.gen(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_c=3):
        '''
            Args:
            - in_c (int) : number of input image channels to disc
        '''
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            convBlock(in_c=in_c, out_c=64, disc=True, use_bn=False, use_act=True, kernel_size=4, stride=2, padding=1),
            convBlock(in_c=64, out_c=128, disc=True, use_bn=True, use_act=True, kernel_size=4, stride=2, padding=1),
            convBlock(in_c=128, out_c=256, disc=True, use_bn=True, use_act=True, kernel_size=4, stride=2, padding=1),
            convBlock(in_c=256, out_c=512, disc=True, use_bn=True, use_act=True, kernel_size=4, stride=2, padding=1),
            convBlock(in_c=512, out_c=1, disc=True, use_bn=False, use_act=False, kernel_size=4, stride=1, padding=0),
            nn.Flatten(),
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.disc(x)
        return out

class convBlock(nn.Module):
    def __init__(self, in_c, out_c, disc=False, use_bn=True, use_act=True, **kwargs):
        '''
            Args:
            - in_c (int) : number of input channels to conv layer
            - out_c (int) : number of output channels from conv layer
            - disc (bool) : flag to use convBlock in gen or disc (activation changes)
            - use_bn (bool) : flag to identify if bn is to be used
            - use_act (bool) : flag to identify if activation is to be used
        '''
        super(convBlock, self).__init__()
        self.use_bn = use_bn
        self.use_act = use_act
        self.conv = nn.ConvTranspose2d(in_c, out_c, bias=not use_bn, **kwargs) if disc==False else nn.Conv2d(in_c, out_c, bias=not use_bn,**kwargs)
        self.bn = nn.BatchNorm2d(out_c) if disc==False else nn.InstanceNorm2d(out_c, affine=True)   
        self.act = nn.ReLU() if disc==False else nn.LeakyReLU(0.2)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out) if self.use_bn else out
        out = self.act(out) if self.use_act else out
        return out
    
def initialize_weights(model):
    '''
        All weights are initialized from a normal distribution with mean=0 and std=0.02.
    '''
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    bs = 16
    img_c = 1
    z_dim = 100

    x_disc = torch.randn((bs, img_c, 64, 64)).cuda()
    x_gen = torch.rand((bs, z_dim, 1, 1)).cuda()  # uniform distribution
    gen = Generator(z_dim=z_dim, img_c=img_c).cuda()
    disc = Discriminator(in_c=img_c).cuda()

    out_gen = gen(x_gen)
    out_disc = disc(x_disc)

    assert out_gen.shape == (bs, img_c, 64, 64), "generator output shape doesn't match"
    assert out_disc.shape == (bs, 1), "discriminator output shape doesn't match"

    summary(gen, (z_dim, 1, 1))
    summary(disc, (img_c, 64, 64))

    print("\n# ---Testing Done--- #\n")


if __name__ == '__main__':
    test()