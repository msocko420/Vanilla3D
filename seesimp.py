import numpy as np
import matplotlib.pyplot as plt

def visualize_samples(epoch):
    sample_path = f'generated_samples/sample_epoch_{100}.npy'
    sample = np.load(sample_path)
    
    # Visualize the samples
    fig, ax = plt.subplots()
    ax.imshow(sample[0, 0, :, :, sample.shape[4] // 2], cmap='gray')  # Display a slice in the middle
    plt.show()

# Specify the epoch of the samples you want to visualize
epoch_to_visualize = 0

# Call the visualize_samples function
visualize_samples(epoch_to_visualize)
