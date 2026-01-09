import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

try:
    df = pd.read_csv('data/HAM10000_metadata.csv')
    image_dir = 'data/all_images/'
    print(f"Success! Found {len(df)} image entries in the CSV.")
    
    sample_id = df['image_id'][0]
    if os.path.exists(os.path.join(image_dir, f"{sample_id}.jpg")):
        print("Verified: Images are in the correct folder.")
    else:
        print("Error: Images not found in data/all_images/")

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        img_id = df['image_id'][i]
        label = df['dx'][i]
        img = Image.open(os.path.join(image_dir, f"{img_id}.jpg"))
        axes[i].imshow(img)
        axes[i].set_title(f"Type: {label}")
        axes[i].axis('off')
    
    print("Opening preview window... (Close it to continue)")
    plt.show()

except Exception as e:
    print(f"Something went wrong: {e}")
