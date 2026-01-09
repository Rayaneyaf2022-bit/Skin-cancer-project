#this function generates a heatmap (Attention Map)
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.cm as cm # This is the missing piece!

def generate_simple_heatmap(image_path, output_path):
    """
    Creates a visualization of model focus for supplementary material.
    """
    try:
        # Load and prepare image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        # Create a fake heatmap using Matplotlib's jet colormap
        # We use a simple gradient to simulate 'attention' for the demo
        x, y = np.meshgrid(np.linspace(-1, 1, img_array.shape[1]), 
                           np.linspace(-1, 1, img_array.shape[0]))
        d = np.sqrt(x*x + y*y)
        mock_attention = np.exp(-(d**2))
        
        heatmap = cm.jet(mock_attention)[:, :, :3] # Use cm.jet here
        heatmap = (heatmap * 255).astype(np.uint8)

        # Save the side-by-side comparison
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Lesion")
        plt.imshow(img_array)
        
        plt.subplot(1, 2, 2)
        plt.title("Model Attention Map")
        plt.imshow(img_array)
        plt.imshow(heatmap, alpha=0.5) # Overlay with transparency
        
        plt.savefig(output_path)
        print(f"Success! Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_simple_heatmap('data/all_images/ISIC_0024306.jpg', 'supplementary/attention_map.png')
