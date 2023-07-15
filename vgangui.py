import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from skimage.measure import profile_line
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QProgressBar, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

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
    def __init__(self, z_dim=100, channels=64, dropout_prob=0.2):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose3d(z_dim, channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),  # Dropout layer added
            nn.ConvTranspose3d(channels * 8, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),  # Dropout layer added
            nn.ConvTranspose3d(channels * 4, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),  # Dropout layer added
            nn.ConvTranspose3d(channels * 2, channels, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),  # Dropout layer added
            nn.ConvTranspose3d(channels, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.gen(input)


class Discriminator(nn.Module):
    def __init__(self, channels=64, dropout_prob=0.2):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv3d(1, channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),  # Dropout layer added
            nn.Conv3d(channels, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),  # Dropout layer added
            nn.Conv3d(channels * 2, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),  # Dropout layer added
            nn.Conv3d(channels * 4, channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob),  # Dropout layer added
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
optimizerD = optim.Adam(netD.parameters(), lr=0.00002, betas=(0.9, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.00002, betas=(0.9, 0.999))

# Setting up directories and other parameters
data_dir = 'C:\\Users\\migue\\shapegan\\data\\vox64'
batch_size = 100
num_epochs = 1000

# Data Loading
dataset = VoxelDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GAN Training")
        self.progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, num_epochs)
        self.progress_bar.setTextVisible(True)
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.start_button)
        layout.addWidget(self.image_label)
        
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.current_epoch = 0
    
    def start_training(self):
        self.timer.start(1000)  # Update progress every 1 second
        self.start_button.setEnabled(False)
        self.train_epoch()
    
    def train_epoch(self):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), 1, device=device, dtype=torch.float32)
            output = netD(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, 100, 1, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(1)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            optimizerG.step()

            writer.add_scalar('Loss/Discriminator', errD.item(), self.current_epoch * len(dataloader) + i)
            writer.add_scalar('Loss/Generator', errG.item(), self.current_epoch * len(dataloader) + i)

            for name, param in netD.named_parameters():
                writer.add_histogram(f'Discriminator/Gradients/{name}', param.grad, self.current_epoch)
                writer.add_histogram(f'Discriminator/Weights/{name}', param, self.current_epoch)
            for name, param in netG.named_parameters():
                writer.add_histogram(f'Generator/Gradients/{name}', param.grad, self.current_epoch)
                writer.add_histogram(f'Generator/Weights/{name}', param, self.current_epoch)

        self.current_epoch += 1

        # Update generated sample image
        if self.current_epoch % 10 == 0:
            self.update_generated_image()
    
    def update_progress(self):
        self.progress_bar.setValue(self.current_epoch)
        if self.current_epoch >= num_epochs:
            self.timer.stop()
            self.start_button.setEnabled(True)
    
    def update_generated_image(self):
        with torch.no_grad():
            fake = netG(torch.randn(batch_size, 100, 1, 1, 1, device=device)).detach().cpu()
            fake_image = make_grid(fake[:16], nrow=4, normalize=True)
            fake_image = fake_image.permute(1, 2, 0).numpy()
            fake_image = (fake_image * 255).astype(np.uint8)
            h, w, ch = fake_image.shape
            bytes_per_line = ch * w
            q_image = QImage(fake_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap(q_image)
            self.image_label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        writer.close()
        event.accept()

# Create the application and main window
app = QApplication([])
window = MainWindow()
window.show()

app.exec_()
