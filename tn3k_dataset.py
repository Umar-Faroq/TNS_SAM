import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class TN3KDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Resize both image and mask to the same size
        img = img.resize((256, 256))
        mask = mask.resize((256, 256))

        if self.transform:
            img = self.transform(img)

        # Convert mask to tensor and make sure it's the same shape
        mask = np.array(mask)
        mask = (mask > 127).astype(np.uint8)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # shape: [1, H, W]

        return img, mask, os.path.basename(self.image_paths[idx])
