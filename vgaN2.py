import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from skimage.measure import profile_line
import matplotlib.pyplot as plt

# Define a function to save the generated samples
def save_sample(sample, epoch):
    np.save(f'generated_samples/sample_epoch_{epoch}.npy', sample)

# VoxelDataset class
class VoxelDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(file_path)  # Load numpy array
        data = torch.from_numpy(data).float()  # Convert to PyTorch tensor
        data = data.unsqueeze(0)  # Add extra dimension for single-channel
        data = data.unsqueeze(0)  # Add extra dimension for batch size
        return data

class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=64, dropout_prob=0.1):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose3d(z_dim, channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose3d(channels * 8, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose3d(channels * 4, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose3d(channels * 2, channels, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose3d(channels, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.gen(input)


class Discriminator(nn.Module):
    def __init__(self, channels=64, dropout_prob=0.3):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv3d(1, channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv3d(channels, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv3d(channels * 2, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv3d(channels * 4, channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv3d(channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.disc(input)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Initialize the tensorboard writer
writer = SummaryWriter()

# Device selection, model definition, and initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

# Criterion and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.00002, betas=(0.9, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=0.00002, betas=(0.9, 0.9))

# Setting up directories and other parameters
data_dir = 'C:\\Users\\migue\\shapegan\\data\\vox64'
batch_size = 64
num_epochs = 500

# Data Loading
dataset = VoxelDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the last saved model weights if available
generator_file = 'generator.pth'
discriminator_file = 'discriminator.pth'
if os.path.exists(generator_file) and os.path.exists(discriminator_file):
    netG.load_state_dict(torch.load(generator_file))
    netD.load_state_dict(torch.load(discriminator_file))

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real = data[0].to(device)
        b_size = real.size(0)
        label = torch.full((b_size,), 1, device=device, dtype=torch.float32)
        # Forward pass real batch through D
        output = netD(real).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 100, 1, 1, 1, device=device)
        # Generate fake voxel set
        fake = netG(noise)
        label.fill_(0)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)  # Apply gradient clipping
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(1)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)  # Apply gradient clipping
        optimizerG.step()

        # Log scalar values (Losses)
        writer.add_scalar('Loss/Discriminator', errD.item(), epoch * len(dataloader) + i)
        writer.add_scalar('Loss/Generator', errG.item(), epoch * len(dataloader) + i)

        # Log gradients and weights
        for name, param in netD.named_parameters():
            writer.add_histogram(f'Discriminator/Gradients/{name}', param.grad, epoch)
            writer.add_histogram(f'Discriminator/Weights/{name}', param, epoch)
        for name, param in netG.named_parameters():
            writer.add_histogram(f'Generator/Gradients/{name}', param.grad, epoch)
            writer.add_histogram(f'Generator/Weights/{name}', param, epoch)

    print(f"[{epoch+1}/{num_epochs}] Loss_D: {errD.item()} Loss_G: {errG.item()}")

    if epoch % 10 == 0:  # Save and log samples every 10 epochs
        # Generate samples and save them
        with torch.no_grad():
            fake = netG(torch.randn(batch_size, 100, 1, 1, 1, device=device)).detach().cpu()

            if not os.path.exists('generated_samples'):
                os.makedirs('generated_samples')

            save_sample(fake.numpy(), epoch)

            # Maximum Intensity Projection
            mip_fake = fake.numpy().max(axis=2)  # MIP along the third axis. Adjust the axis as needed.
            mip_fake = torch.from_numpy(mip_fake)  # Convert back to tensor for make_grid

            # Log MIP of images (fake samples)
            img_grid = make_grid(mip_fake[:32].reshape(-1, 1, 64, 64), nrow=8, normalize=True)
            writer.add_image('Generated Images', img_grid, epoch)

            # Generate and save matplotlib figures
            fig, ax = plt.subplots()
            ax.imshow(fake.numpy()[0, 0, :, :, fake.shape[3] // 2], cmap='gray')  # Display a slice in the middle
            plt.savefig(f'generated_samples/sample_epoch_{epoch}.png')  # Save the figure
            plt.close(fig)  # Close the figure to avoid warning

# Save model weights at the end of training
torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')

writer.close()