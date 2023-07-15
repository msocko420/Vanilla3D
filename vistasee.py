import numpy as np
import pyvista as pv

# Load the saved numpy sample
sample_file = 'generated_samples/sample_epoch_100.npy'  # Replace with the path to your saved sample
sample = np.load(sample_file)

# Create PyVista image data and assign the sample as point data
image_data = pv.UniformGrid()
image_data.dimensions = (sample.shape[4], sample.shape[3], sample.shape[2])
image_data.origin = (0, 0, 0)
image_data.spacing = (1, 1, 1)
image_data.point_data['Sample'] = sample[0, 0].flatten(order='F')  # Reshape the sample array

# Plot the image data using PyVista
p = pv.Plotter(notebook=True)
p.add_volume(image_data, cmap='coolwarm')
p.show()

output_file = 'generated_image.png'
p.screenshot(output_file)
print(f"Image saved to {output_file}")
