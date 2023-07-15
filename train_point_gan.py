import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from torch.utils.tensorboard import SummaryWriter
from datasets import PointDataset
from model.point_sdf_net import PointNet, SDFGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import io
from PIL import Image

# Output Directory
output_dir = 'C:/users/migue/shapegan/outputPOINT'
os.makedirs(output_dir, exist_ok=True)

LATENT_SIZE = 128
GRADIENT_PENALITY = 10
HIDDEN_SIZE = 256
NUM_LAYERS = 8
NORM = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
G = SDFGenerator(LATENT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NORM, dropout=0.0)
D = PointNet(out_channels=1)
G, D = G.to(device), D.to(device)
G_optimizer = RMSprop(G.parameters(), lr=0.0001)
D_optimizer = RMSprop(D.parameters(), lr=0.0001)

# Dataset instantiation
dataset_path = "C:/users/migue/shapegan/data/sdf"
filenames = [f for f in os.listdir(dataset_path)]
dataset = PointDataset(dataset_path, filenames)

configuration = [  # num_points, batch_size, epochs
    (1024, 82, 1000),
    (2048, 52, 700),
    (4096, 52, 500),
    (8192, 24, 400),
    (16384, 12, 600),
    (32768, 6, 900),
]

# Convert plot to image
def plot_to_image(figure):
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to PIL Image
    img = Image.open(buf).convert('RGB')
    image_np = np.array(img)

    return image_np

# Plot points
def plot_3d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    return fig

# Save generated samples
def save_generated_samples(fake_points, epoch, num_steps):
    if epoch % 100 == 0:
        np.save(osp.join(output_dir, f"generated_samples_{epoch}_{num_steps}.npy"), fake_points)
    if epoch % 10 == 0:
        fig = plot_3d_points(fake_points)
        plt.savefig(osp.join(output_dir, f"generated_samples_{epoch}_{num_steps}.png"))
        plt.close(fig)

# Save the model
def save_model(model, model_name):
    model_path = osp.join(output_dir, model_name)
    torch.save(model.state_dict(), model_path)

# TensorBoard writer
writer = SummaryWriter()

num_steps = 0
best_loss = float('inf')
for num_points, batch_size, epochs in configuration:
    dataset.num_points = num_points
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_D_loss = 0
        total_G_loss = 0
        total_gp = 0

        for uniform in loader:
            num_steps += 1

            uniform = uniform.to(device)
            u_pos, u_dist = uniform[..., :3], uniform[..., 3:]

            # Train the discriminator
            D_optimizer.zero_grad()
            z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
            fake = G(u_pos, z)
            out_real = D(u_pos, u_dist)
            out_fake = D(u_pos, fake)
            D_loss = out_fake.mean() - out_real.mean()

            # Gradient penalty
            alpha = torch.rand((uniform.size(0), 1, 1), device=device)
            interpolated = alpha * u_dist + (1 - alpha) * fake
            interpolated.requires_grad_(True)
            out = D(u_pos, interpolated)
            grad = torch.autograd.grad(out, interpolated, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_norm = grad.view(grad.size(0), -1).norm(dim=-1, p=2)
            gp = GRADIENT_PENALITY * ((grad_norm - 1).pow(2).mean())

            # Final discriminator loss
            loss = D_loss + gp
            loss.backward()
            D_optimizer.step()

            total_loss += loss.item()
            total_D_loss += D_loss.item()
            total_gp += gp.item()

            # Train the generator every 5 steps
            if num_steps % 5 == 0:
                G_optimizer.zero_grad()
                z = torch.randn(uniform.size(0), LATENT_SIZE, device=device)
                fake = G(u_pos, z)
                out_fake = D(u_pos, fake)
                G_loss = -out_fake.mean()
                G_loss.backward()
                G_optimizer.step()

                total_G_loss += G_loss.item()

        avg_loss = total_loss / len(loader)
        avg_d_loss = total_D_loss / len(loader)
        avg_g_loss = total_G_loss / len(loader)
        avg_gp = total_gp / len(loader)

        print('Num points: {}, Epoch: {:03d}, Total Loss: {:.6f}, D Loss: {:.6f}, G Loss: {:.6f}, GP: {:.6f}'.format(
            num_points, epoch, avg_loss, avg_d_loss, avg_g_loss, avg_gp))

        writer.add_scalar('Total Loss', avg_loss, num_steps)
        writer.add_scalar('Discriminator Loss', avg_d_loss, num_steps)
        writer.add_scalar('Generator Loss', avg_g_loss, num_steps)
        writer.add_scalar('Gradient Penalty', avg_gp, num_steps)

        if avg_loss < best_loss:
            save_model(G, "best_generator.pt")
            save_model(D, "best_discriminator.pt")
            best_loss = avg_loss

        # visualize every 50 epochs
        if epoch % 50 == 0:
            with torch.no_grad():
                fake_points = fake.detach().cpu().squeeze().numpy()
                save_generated_samples(fake_points, epoch, num_steps)

                fig = plot_3d_points(fake_points)
                img = plot_to_image(fig)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                writer.add_image('Generated Samples', img_tensor, num_steps)

writer.close()