from tn3k_dataset import TN3KDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from segment_anything import sam_model_registry
from sklearn.model_selection import train_test_split
import os
from glob import glob

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Small value to avoid division by zero

    def forward(self, preds, targets):
        """Computes Dice Loss.
        Args:
            preds: Predicted segmentation mask (logits before sigmoid activation).
            targets: Ground truth mask (binary).
        Returns:
            Dice loss value.
        """
        preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
        intersection = (preds * targets).sum(dim=(2, 3))  # Compute intersection
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))  # Compute union
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)  # Dice Score
        dice_loss = 1 - dice_coeff.mean()  # Convert to loss
        return dice_loss

# Get all image and mask paths
image_dir = "/home/dilab/ext_drive/Thyroid_Nodule_segmentation/Thyroid_Dataset/TN3K/trainval-image"
mask_dir = "/home/dilab/ext_drive/Thyroid_Nodule_segmentation/Thyroid_Dataset/TN3K/trainval-mask"

all_images = sorted(glob(os.path.join(image_dir, "*")))
all_masks = sorted(glob(os.path.join(mask_dir, "*")))

# Split into 80% train, 20% val
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    all_images, all_masks, test_size=0.2, random_state=42
)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = TN3KDataset(train_imgs, train_masks, transform=transform)
val_dataset   = TN3KDataset(val_imgs, val_masks, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

from segment_anything import sam_model_registry

model = sam_model_registry["vit_b"](checkpoint="work_dir/medsam_vit_b.pth")
model = model.to("cuda")
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
dice_loss = DiceLoss()
for epoch in range(5):
    epoch_loss = 0
    for images, masks, _ in train_loader:
        images = images.cuda()
        masks = masks.cuda()

        preds = model(images)  # assumes model outputs logits or probabilities
        loss = dice_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/5, Avg Loss: {avg_loss:.4f}")