import torch
import torch.nn as nn
from torchsummary import summary


# generator and discriminator classes below; both MLPs
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        '''
            Args:
            - z_dim (int) : random vector (noise) dim
            - img_dim (int) : length when img is unrolled
        '''
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.gen(x)
        return out

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        '''
            Args:
            - img_dim (int) : length when img is unrolled
        '''
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 4096),
            nn.LeakyReLU(0.01),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid()    # to classify fake and real
        )

    def forward(self, x):
        out = self.disc(x)
        return out


def test():
    bs = 64
    in_c = 3
    z_dim = 64
    img_dim = 64*64*3   # rgb imgs

    x = torch.randn((bs, in_c, 64, 64))
    x_disc = x.view(-1, img_dim).cuda()
    x_gen = torch.randn((bs, z_dim)).cuda()
    gen = Generator(z_dim=z_dim, img_dim=img_dim).cuda()
    disc = Discriminator(img_dim=img_dim).cuda()

    out_gen = gen(x_gen)
    out_disc = disc(x_disc)

    assert out_gen.shape == (bs, img_dim), "generator output shape doesn't match"
    assert out_disc.shape == (bs, 1), "discriminator output shape doesn't match"

    summary(gen, (in_c, 64, 64))
    summary(disc, (1, in_c*64*64))

    print("\n# ---Testing Done--- #\n")


if __name__ == '__main__':
    test()