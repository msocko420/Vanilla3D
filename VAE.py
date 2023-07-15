import os
import torch
import numpy as np
from torch import nn
from torch.optim import AdamW
from torch.nn import LeakyReLU, LayerNorm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

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

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.activation = LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        self.norm = LayerNorm([x.size(1), x.size(2), x.size(3), x.size(4)]).to(x.device)
        return self.activation(self.norm(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.activation = LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        self.norm = LayerNorm([x.size(1), x.size(2), x.size(3), x.size(4)]).to(x.device)
        return self.activation(self.norm(x))


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder layers
        self.enc1 = EncoderBlock(1, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.enc5 = EncoderBlock(512, 1024)

        # Latent layers
        self.fc_mu = nn.Linear(8192, 250)
        self.fc_var = nn.Linear(8192, 250)

        # Decoder layers
        self.dec1 = DecoderBlock(250, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        self.dec5 = DecoderBlock(64, 32)
        self.dec6 = DecoderBlock(32, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        # Flatten encoder output
        x = x.view(x.size(0), -1)

        # Get latent variables
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        z = self.reparameterize(mu, logvar)

        # Decoder
        z = z.view(z.size(0), z.size(1), 1, 1, 1)
        z = self.dec1(z)
        z = self.dec2(z)
        z = self.dec3(z)
        z = self.dec4(z)
        z = self.dec5(z)
        reconstruction = torch.sigmoid(self.dec6(z))

        return reconstruction, mu, logvar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 80000
batch_size = 100
learning_rate = 0.0002

# Create VAE
vae = VAE().to(device)
optimizer = AdamW(vae.parameters(), lr=learning_rate)

# Load Data
data_dir = 'C:\\Users\\migue\\shapegan\\data\\vox64'
dataset = VoxelDataset(data_dir)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

writer = SummaryWriter()
# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0

    for i, real_data in enumerate(data_loader):
        real_data = real_data.to(device)

        # Forward pass
        reconstruction, mu, logvar = vae(real_data)

        # Loss
        recon_loss = nn.functional.binary_cross_entropy(reconstruction, real_data, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss components
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kld_loss += kld_loss.item()

    # Average loss components over all batches
    avg_loss = total_loss / len(data_loader)
    avg_recon_loss = total_recon_loss / len(data_loader)
    avg_kld_loss = total_kld_loss / len(data_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, Recon Loss: {avg_recon_loss}, KLD Loss: {avg_kld_loss}')

    # Log the loss and its components
    writer.add_scalar('Loss/Total', avg_loss, epoch)
    writer.add_scalar('Loss/Reconstruction', avg_recon_loss, epoch)
    writer.add_scalar('Loss/KLD', avg_kld_loss, epoch)

    # Log histograms of model parameters
    for name, param in vae.named_parameters():
        writer.add_histogram(name, param, epoch)

    # Every 100 epochs, log generated samples
    if epoch % 100 == 0:
        with torch.no_grad():
            # Assume z is your latent vector and get a sample from the normal distribution
            z = torch.randn(batch_size, 250).to(device)
            out = vae.decoder(z).cpu()
            # Add the sample to TensorBoard
            writer.add_images('Generated Samples', out, epoch)

    # Visualize the distributions of the learned mu and logvar
    writer.add_histogram('mu', mu, epoch)
    writer.add_histogram('logvar', logvar, epoch)

    # Save the trained model every 100 epochs
    if epoch % 100 == 0:
        torch.save(vae.state_dict(), f'vae_epoch_{epoch}.pth')

# Save the trained model
torch.save(vae.state_dict(), 'vae.pth')

# Close the SummaryWriter
writer.close()


