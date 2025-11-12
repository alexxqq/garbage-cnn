"""
Data Loader for Garbage Classification Dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import torchvision.transforms as transforms

class GarbageDataset(Dataset):
    """Dataset class for garbage classification"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Class names
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load images and labels
        for cls in self.classes:
            cls_path = self.data_dir / cls
            if not cls_path.exists():
                continue
            
            for img_path in cls_path.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[cls])
            
            for img_path in cls_path.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(data_root="data/processed", batch_size=32, img_size=64):
    """Create data loaders for train, val, and test sets"""
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = GarbageDataset(
        os.path.join(data_root, "train"),
        transform=train_transform
    )
    
    val_dataset = GarbageDataset(
        os.path.join(data_root, "val"),
        transform=val_test_transform
    )
    
    test_dataset = GarbageDataset(
        os.path.join(data_root, "test"),
        transform=val_test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

