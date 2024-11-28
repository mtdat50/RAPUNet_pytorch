import os
import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset


def glob_paths(data_folder):
    """
    Args:
        data_folder: Path to the folder containing images/ and masks/.

    Returns:
        image_paths, mask_paths.
    """
    path_parts = [data_folder, "images", "*"]
    image_path = os.path.join(*path_parts)
    mask_path = image_path.replace("images", "masks")

    return sorted(glob(image_path)), sorted(glob(mask_path))


class FolderDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Args:
            data_folder: Path to the folder containing image/ and mask/.
            transform: albumentations.Compose, the augmentation pipeline.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"), dtype=np.float32) / 255
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"), dtype=np.float32) / 255

        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask

# Example usage
# if __name__ == "__main__":
#     # Paths to dataset folders
#     image_folder = "path/to/images"
#     mask_folder = "path/to/masks"

#     # Create dataset
#     dataset = FolderDataset(image_folder, mask_folder, transform=aug_train)

#     # Create DataLoader
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#     # Iterate through the DataLoader
#     for images, masks in dataloader:
#         print(f"Images batch shape: {images.shape}")
#         print(f"Masks batch shape: {masks.shape}")
#         break
