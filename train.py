"""
High-Level CNN Training for Garbage Classification
Uses Transfer Learning with ResNet18 (pre-trained on ImageNet)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Configuration
IMG_SIZE = 224  # ResNet standard input size
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
DATA_ROOT = "data/processed"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GarbageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images, self.labels = [], []
        for i, cls in enumerate(CLASSES):
            for img_path in (self.data_dir / cls).glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(i)
            for img_path in (self.data_dir / cls).glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(i)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Model: ResNet50 with Transfer Learning
def create_model():
    """Create ResNet50 model with pre-trained ImageNet weights"""
    # Load pre-trained ResNet50 (larger, more powerful than ResNet18)
    model = models.resnet50(weights='IMAGENET1K_V2')
    # Replace final layer for our 6 garbage classes
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model

# Data
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(GarbageDataset(f"{DATA_ROOT}/train", transform_train), 
                         batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(GarbageDataset(f"{DATA_ROOT}/val", transform_val), 
                       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

if __name__ == '__main__':
    # Training
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0
    print(f"Training ResNet50 on {device}...")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = train_correct = train_total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        # Validate
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        print(f"Train: {train_loss/len(train_loader):.4f} ({train_acc:.1f}%) | "
              f"Val: {val_loss/len(val_loader):.4f} ({val_acc:.1f}%)")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_resnet50.pth")
            print(f"✓ Saved (val_acc: {val_acc:.1f}%)")
        
        scheduler.step()
    
    print(f"\n✅ Done! Best val accuracy: {best_acc:.1f}%")

