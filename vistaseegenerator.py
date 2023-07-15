import torch
from torch import nn
from torchvision.utils import make_grid
import numpy as np
import pyvista as pv
from pyvista import examples

class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=64, dropout_prob=0.2):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose3d(z_dim, channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose3d(channels * 8, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose3d(channels * 4, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose3d(channels * 2, channels, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),
            nn.ConvTranspose3d(channels, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.gen(input)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)

# Load the generator's weights
generator_file = 'generator.pth'
if torch.cuda.is_available():
    netG.load_state_dict(torch.load(generator_file))
else:
    netG.load_state_dict(torch.load(generator_file, map_location=torch.device('cpu')))
netG.eval()

# Generate a sample
z_dim = 100
sample = netG(torch.randn(1, z_dim, 1, 1, 1, device=device)).detach().cpu().numpy()[0, 0, :, :, :]

# Create PyVista uniform grid from the generated sample
grid = pv.UniformGrid()
grid.dimensions = sample.shape
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.point_arrays['Generated Samples'] = sample.flatten(order='F')

# Create PyVista volume
volume = pv.wrap(grid)

# Plot the volume using PyVista
p = pv.Plotter(notebook=True)
p.add_volume(volume, cmap='coolwarm')
p.show()
