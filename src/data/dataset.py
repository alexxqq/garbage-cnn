"""PyTorch Dataset classes"""
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Callable
from torchvision import transforms

class GarbageDataset(Dataset):
    """Dataset for garbage classification"""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        classes: Optional[list] = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = classes or ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
        self.images = []
        self.labels = []
        
        # Load images
        for i, cls in enumerate(self.classes):
            cls_dir = self.data_dir / cls
            if not cls_dir.exists():
                continue
            
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img_path in cls_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(i)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple:
        img = Image.open(self.images[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[idx]

def get_transforms(
    image_size: int = 224,
    is_training: bool = False,
    augmentation_config: Optional[dict] = None
) -> transforms.Compose:
    """Get data transforms"""
    if augmentation_config is None:
        augmentation_config = {}
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(augmentation_config.get('horizontal_flip', 0.5)),
            transforms.RandomRotation(augmentation_config.get('rotation', 10)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

