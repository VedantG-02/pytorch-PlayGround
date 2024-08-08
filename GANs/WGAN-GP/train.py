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


torch.manual_seed(42) # reproducibility purpose


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
critic = Discriminator(in_c=3).to(DEVICE)
gen = Generator(z_dim=Z_DIM, img_c=3).to(DEVICE)
initialize_weights(gen)
initialize_weights(critic)
opt_critic = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.0, 0.9))
opt_gen = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.0, 0.9))


# fixed noise to see results of while training generator (on tensorboard)
fixed_noise = torch.randn((64, Z_DIM, 1, 1)).to(DEVICE)


# tensorboard writers
BOARD_PATH = "runs"
writer_fake = SummaryWriter(f"WGAN-GP/tensorboard/{BOARD_PATH}/fake")
writer_real = SummaryWriter(f"WGAN-GP/tensorboard/{BOARD_PATH}/real")
step = 0


# gp code
def gradient_penalty(critic, real, fake, device="cpu"):
    epsilon = torch.rand((real.shape[0], 1, 1, 1)).repeat(1, real.shape[1], real.shape[2], real.shape[3]).to(device)
    interpolated_imgs = epsilon*real + (1-epsilon)*fake
    critic_scores = critic(interpolated_imgs)

    gradient = torch.autograd.grad(
        inputs=interpolated_imgs,
        outputs=critic_scores,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty


# training
critic.train()
gen.train()
print("\nTraining is starting...\n")
for epoch in range(30):
    for batch_id, (real, _) in enumerate(dataloader):
        # print(batch_id+1)
        real = real.to(DEVICE)
        batch_size = real.shape[0]

        # train critic : min(E(f(fake)) - E(f(real)) + lambda*gp(interpolated_imgs))
        for i in range(5): # n_critic=5; not experimenting with this
            noise = torch.randn(batch_size, Z_DIM, 1, 1).to(DEVICE)
            fake = gen(noise)
            critic_real = critic(real).view(-1)
            critic_fake = critic(fake.detach()).view(-1)
            grad_pen = gradient_penalty(critic, real, fake, device=DEVICE) # gradient penalty 
            loss_critic = torch.mean(critic_fake) - torch.mean(critic_real) + 10*(grad_pen) # lambda=10
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # train generator : max(E(f(fake)))
        output = critic(fake).view(-1)
        loss_gen = -1*torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if (batch_id+1) % (len(dataloader)//2) == 0:
            print(f"Epoch [{epoch+1:>2}/30], Batch {batch_id+1}/{len(dataloader)}, Loss D: {loss_critic:.5f}, Loss G: {loss_gen:.5f}")

    # log at each epoch end
    with torch.no_grad():
        fake = gen(fixed_noise)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(real, normalize=True)

        writer_fake.add_image("CELEB Fake Images", img_grid_fake, global_step=step)
        writer_real.add_image("CELEB Real Images", img_grid_real, global_step=step)
        
        step += 1
    print()

if not os.path.isdir(os.path.join('WGAN-GP', 'models')):
    os.mkdir(os.path.join('WGAN-GP', 'models'))
torch.save(gen, 'WGAN-GP/models/generator.pth')
torch.save(critic, 'WGAN-GP/models/critic.pth')