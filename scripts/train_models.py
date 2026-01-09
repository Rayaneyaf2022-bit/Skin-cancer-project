import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTForImageClassification
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# 1.defining the CNN
def get_resnet():
    model = models.resnet50(pretrained=True)
    # Adjust for 7 skin cancer classes
    model.fc = nn.Linear(model.fc.in_features, 7)
    return model.to(device)

# 2.defining the ViT
def get_vit():
    model = ViTForImageClassification.from_pretrained(
        "models/vit-local/",
        num_labels=7,
        ignore_mismatched_sizes=True
    )
    return model.to(device)

print("The Models are ready")

import torch.optim as optim
import pandas as pd
import os
# 3.the training function
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Note: ViT outputs are slightly different than ResNet
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

print("Training logic is set, Ready")

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 4.Custom Dataset Class
class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        # Map labels (nv, mel, etc.) to numbers (0-6)
        self.label_map = {val: i for i, val in enumerate(df['dx'].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        label = self.label_map[self.df.iloc[idx]['dx']]
        
        if self.transform:
            image = self.transform(image)
        return image, label

# 5.Preprocessing required for ViT and ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Data Bridge ready, We can now feed the images.")

if __name__ == "__main__":
    df = pd.read_csv('data/HAM10000_metadata.csv')
    
    # now we run on everything
    dataset = SkinDataset(df, 'data/all_images/', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True) # M4 can handle 32!

    resnet = get_resnet()
    vit = get_vit()
    
    criterion = nn.CrossEntropyLoss()
    optimizer_res = optim.Adam(resnet.parameters(), lr=0.0001)
    optimizer_vit = optim.Adam(vit.parameters(), lr=0.00001)

    print("\n--- Starting Full Comparison (10,015 Images) ---")
    
    res_loss = train_one_epoch(resnet, loader, optimizer_res, criterion)
    print(f"Final ResNet-50 Loss: {res_loss:.4f}")

    vit_loss = train_one_epoch(vit, loader, optimizer_vit, criterion)
    print(f"Final ViT Loss: {vit_loss:.4f}")

# Saving the models
torch.save(resnet.state_dict(), 'models/resnet50_skin.pth')
torch.save(vit.state_dict(), 'models/vit_skin.pth')
print("Models Saved Successfully")
