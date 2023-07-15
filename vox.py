import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm, clip_grad_norm_
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import wandb

# Initialize wandb
wandb.init(project="3D_GAN")

n_critic = 1  # Number of critic iterations per generator iteration
grad_clip_val = 0.1  # Value for gradient clipping

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
        return data


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv3d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv3d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv3d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.xavier_uniform_(self.value_conv.weight)

    def forward(self, x):
        batch_size, C, width, height, depth = x.size()
        query = self.query_conv(x).view(batch_size, -1, width*height*depth).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width*height*depth)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch_size, -1, width*height*depth)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height, depth)
        out = self.gamma*out + x
        return out, attention

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=1, dim=64, out_conv_channels=1024):
        super(Discriminator, self).__init__()
        conv1_channels = int(out_conv_channels / 32)
        conv2_channels = int(out_conv_channels / 16)
        conv3_channels = int(out_conv_channels / 8)
        conv4_channels = int(out_conv_channels / 4)
        conv5_channels = int(out_conv_channels / 2)
        self.out_conv_channels = out_conv_channels
        self.out_dim = int(dim / 64)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=conv1_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=conv3_channels, out_channels=conv4_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(conv4_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=conv4_channels, out_channels=conv5_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(conv5_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=conv5_channels, out_channels=out_conv_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Linear(out_conv_channels * self.out_dim * self.out_dim * self.out_dim, 1)
        self.self_attention = SelfAttention(conv5_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x, _ = self.self_attention(self.conv5(x))
        x = self.conv6(x)
        x = x.view(-1, self.out_conv_channels * self.out_dim * self.out_dim * self.out_dim)
        x = self.out(x)
        return x
    
class Generator(torch.nn.Module):
    def __init__(self, in_channels=1024, out_dim=64, out_channels=1, noise_dim=250, activation="sigmoid"):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.in_dim = int(out_dim / 64)
        conv1_out_channels = int(self.in_channels / 2.0)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)
        conv4_out_channels = int(conv3_out_channels / 2)
        conv5_out_channels = int(conv4_out_channels / 2)

        self.linear = torch.nn.Linear(noise_dim, in_channels * self.in_dim * self.in_dim * self.in_dim)
        self.drop = nn.Dropout(0.2)

        self.conv1 = nn.Sequential(nn.ConvTranspose3d(in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False), nn.BatchNorm3d(conv1_out_channels), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.ConvTranspose3d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False), nn.BatchNorm3d(conv2_out_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.ConvTranspose3d(in_channels=conv2_out_channels, out_channels=conv3_out_channels, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False), nn.BatchNorm3d(conv3_out_channels), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.ConvTranspose3d(in_channels=conv3_out_channels, out_channels=conv4_out_channels, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False), nn.BatchNorm3d(conv4_out_channels), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(in_channels=conv4_out_channels, out_channels=conv5_out_channels, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False), nn.BatchNorm3d(conv5_out_channels), nn.ReLU(inplace=True))

        if activation == "tanh":
            self.out = nn.Sequential(nn.ConvTranspose3d(in_channels=conv5_out_channels, out_channels=out_channels, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False), nn.Tanh())
        else:
            self.out = nn.Sequential(nn.ConvTranspose3d(in_channels=conv5_out_channels, out_channels=out_channels, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.in_channels, self.in_dim, self.in_dim, self.in_dim)
        x = self.drop(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)
        return x

# Settings
batch_size = 70
noise_dim = 250
num_epochs = 1500
min_d_loss = np.inf  # This will store the lowest discriminator loss
min_g_loss = np.inf  # This will store the lowest generator loss
OUTPUT_DIR = "C:\\Users\\migue\\shapegan\\outputnew25"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Watch the models
wandb.watch([generator, discriminator], log="all")

# Load previous training model weights for generator, if exist
best_model_path = f"{OUTPUT_DIR}/best_model.pt"
if os.path.isfile(best_model_path):
    checkpoint = torch.load(best_model_path)
    generator.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded previous generator weights from {best_model_path}")

# Load previous training model weights for discriminator, if exist
best_discriminator_model_path = f"{OUTPUT_DIR}/best_discriminator_model.pt"
if os.path.isfile(best_discriminator_model_path):
    checkpoint = torch.load(best_discriminator_model_path)
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded previous discriminator weights from {best_discriminator_model_path}")

# Optimizers
optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=0.00008)
optimizer_G = torch.optim.AdamW(generator.parameters(), lr=0.00011)

# Setting up directories
data_dir = 'C:\\Users\\migue\\shapegan\\data\\vox64'

# Data Loading
dataset = VoxelDataset(data_dir)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Noise Sample function
def sample_noise(num_samples):
    return torch.randn(num_samples, noise_dim, device=device)

#Training Loop
for epoch in range(num_epochs):
    for i, real_data in enumerate(data_loader):
        real_data = real_data.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Generate fake data
        noise = sample_noise(real_data.size(0))
        fake_data = generator(noise)

        # Calculate loss for real and fake data
        real_pred = discriminator(real_data)
        fake_pred = discriminator(fake_data.detach())
       
        # Hinge loss for Discriminator
        real_loss = torch.mean(nn.ReLU()(1. - real_pred))
        fake_loss = torch.mean(nn.ReLU()(1. + fake_pred))
        d_loss = (real_loss + fake_loss) / 2

        # Update Discriminator
        d_loss.backward()
        clip_grad_norm_(discriminator.parameters(), grad_clip_val)  # Apply gradient clipping
        optimizer_D.step()
        # Log discriminator loss to wandb
        wandb.log({"D_loss": d_loss.item()})

        # Train Generator every n_critic steps
        if i % n_critic == 0:
            optimizer_G.zero_grad()

            # Generate fake data
            noise = sample_noise(batch_size)
            fake_data = generator(noise)

            # Calculate Generator loss
            pred = discriminator(fake_data)
            g_loss = -torch.mean(pred)

            # Update Generator
            g_loss.backward()
            clip_grad_norm_(generator.parameters(), grad_clip_val)  # Apply gradient clipping
            optimizer_G.step()

        # Log generator loss to wandb
        wandb.log({"G_loss": g_loss.item()})

    # Print losses
    print(f"Epoch: {epoch}, D_loss: {d_loss.item()}, G_loss: {g_loss.item()}")

    if epoch % 10 == 0:
        torch.save(generator.state_dict(), f"{OUTPUT_DIR}/generator_{epoch}.pt")
        torch.save(discriminator.state_dict(), f"{OUTPUT_DIR}/discriminator_{epoch}.pt")
        # Save models to wandb
        wandb.save(f"{OUTPUT_DIR}/generator_{epoch}.pt", base_path=OUTPUT_DIR)
        wandb.save(f"{OUTPUT_DIR}/discriminator_{epoch}.pt", base_path=OUTPUT_DIR)

    # Save visualization of generated voxel data every 20 epochs
    if epoch % 20 == 0:
        output_file = f"{OUTPUT_DIR}/generator_output_epoch_{epoch}.npy"
        np.save(output_file, fake_data.detach().cpu().numpy())
        # Log voxel data as images
        # Select a slice to visualize, for example the central slice
        fake_data_slice = fake_data[:, :, :, fake_data.size(2) // 2]
        # Rescale fake_data from [-1, 1] to [0, 1] and convert it to a grid image
        img_grid = make_grid(0.5 * fake_data_slice[:16].detach() + 0.5).cpu()
        wandb.log({"fake_voxel_samples": [wandb.Image(img_grid, caption=f"Generated voxel epoch {epoch}")]})

# End the run
wandb.finish()
