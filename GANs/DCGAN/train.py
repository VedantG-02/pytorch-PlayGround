import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator, initialize_weights
import os


# torch.manual_seed(42) # reproducibility purpose


# constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 100
IMG_SIZE = 64


# dataset and dataloader
transforms_ = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
dataset = datasets.ImageFolder(root="data/celeb_faces_data/", transform=transforms_)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# model, optim initialization
disc = Discriminator(in_c=3).to(DEVICE)
gen = Generator(z_dim=Z_DIM, img_c=3).to(DEVICE)
initialize_weights(gen)
initialize_weights(disc)
opt_disc = optim.Adam(disc.parameters(), lr=0.00005, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=0.00005, betas=(0.5, 0.999))


# fixed noise to see results of while training generator (on tensorboard)
fixed_noise = torch.randn((64, Z_DIM, 1, 1)).to(DEVICE)


# loss function and tensorboard writers
loss_fn = nn.BCELoss()
BOARD_PATH = "runs"
writer_fake = SummaryWriter(f"DCGAN/tensorboard/{BOARD_PATH}/fake")
writer_real = SummaryWriter(f"DCGAN/tensorboard/{BOARD_PATH}/real")
step = 0


# training
disc.train()
gen.train()
print("\nTraining is starting...\n")
for epoch in range(50):
    for batch_id, (real, _) in enumerate(dataloader):
        # print(batch_id+1)
        real = real.to(DEVICE)
        batch_size = real.shape[0]

        # train discriminator : max(log(D(x)) + log(1-D(G(z))))
        noise = torch.randn(batch_size, Z_DIM, 1, 1).to(DEVICE)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake.detach()).view(-1)
        disc_loss_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_loss_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (disc_loss_real + disc_loss_fake)/2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # train generator : min(log(1-D(G(z)))) or max(log(D(G(z))))
        output = disc(fake).view(-1)
        loss_gen = loss_fn(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if (batch_id+1) % (len(dataloader)//2) == 0:
            print(f"Epoch [{epoch+1:>2}/50], Batch {batch_id+1}/{len(dataloader)}, Loss D: {loss_disc:.5f}, Loss G: {loss_gen:.5f}")

    # log at each epoch end
    with torch.no_grad():
        fake = gen(fixed_noise)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(real, normalize=True)

        writer_fake.add_image("CELEB Fake Images", img_grid_fake, global_step=step)
        writer_real.add_image("CELEB Real Images", img_grid_real, global_step=step)
        
        step += 1
    print()

if not os.path.isdir(os.path.join('DCGAN', 'models')):
    os.mkdir(os.path.join('DCGAN', 'models'))
torch.save(gen, 'DCGAN/models/generator.pth')
torch.save(disc, 'DCGAN/models/discriminator.pth')