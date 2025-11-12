"""
Training Script for Low-level CNN Garbage Classification
"""
import torch
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from cnn_lowlevel import GarbageCNN, cross_entropy_loss
from data_loader import get_data_loaders

def train_epoch(model, train_loader, learning_rate, device):
    """Train for one epoch"""
    model.set_training(True)
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        predictions = model.forward(images)
        
        # Compute loss
        loss, grad_loss = cross_entropy_loss(predictions, labels)
        total_loss += loss.item()
        
        # Backward pass
        model.backward(grad_loss)
        
        # Update weights
        model.update(learning_rate)
        
        # Compute accuracy
        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, device):
    """Validate the model"""
    model.set_training(False)
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions = model.forward(images)
            
            # Compute loss
            loss, _ = cross_entropy_loss(predictions, labels)
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def save_model(model, path, epoch, accuracy, img_size):
    """Save model state"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    torch.save({
        'epoch': epoch,
        'accuracy': accuracy,
        'img_size': img_size,  # Save img_size for loading
        'conv1_weight': model.conv1.weight,
        'conv1_bias': model.conv1.bias,
        'conv2_weight': model.conv2.weight,
        'conv2_bias': model.conv2.bias,
        'conv3_weight': model.conv3.weight,
        'conv3_bias': model.conv3.bias,
        'conv4_weight': model.conv4.weight,
        'conv4_bias': model.conv4.bias,
        'conv5_weight': model.conv5.weight,
        'conv5_bias': model.conv5.bias,
        'fc1_weight': model.fc1.weight,
        'fc1_bias': model.fc1.bias,
        'fc2_weight': model.fc2.weight,
        'fc2_bias': model.fc2.bias,
    }, path)


def load_model(model, path, device):
    """Load model state"""
    checkpoint = torch.load(path, map_location=device)
    
    model.conv1.weight = checkpoint['conv1_weight']
    model.conv1.bias = checkpoint['conv1_bias']
    model.conv2.weight = checkpoint['conv2_weight']
    model.conv2.bias = checkpoint['conv2_bias']
    model.conv3.weight = checkpoint['conv3_weight']
    model.conv3.bias = checkpoint['conv3_bias']
    model.conv4.weight = checkpoint['conv4_weight']
    model.conv4.bias = checkpoint['conv4_bias']
    model.conv5.weight = checkpoint['conv5_weight']
    model.conv5.bias = checkpoint['conv5_bias']
    model.fc1.weight = checkpoint['fc1_weight']
    model.fc1.bias = checkpoint['fc1_bias']
    model.fc2.weight = checkpoint['fc2_weight']
    model.fc2.bias = checkpoint['fc2_bias']
    
    return checkpoint['epoch'], checkpoint.get('accuracy', 0), checkpoint.get('img_size', 64)


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # Hyperparameters
    num_epochs = 20
    # num_epochs = 1
    learning_rate = 0.001
    #learning_rate = 0.01
    batch_size = 32
    #batch_size = 6
    img_size = 64
    # img_size = 16
    data_root = "data/processed"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if data exists
    if not os.path.exists(data_root):
        print(f"Data directory {data_root} not found!")
        print("Please run: python download_dataset.py")
        return
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        data_root, batch_size, img_size
    )
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model
    model = GarbageCNN(num_classes=len(classes), img_size=img_size)
    model = model.to(device)
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0
    
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, learning_rate, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "models/best_model.pth", epoch, val_acc, img_size)
            print(f"Saved best model with val accuracy: {val_acc:.2f}%")
        
        # Learning rate decay
        if epoch % 5 == 0:
            learning_rate *= 0.5
            print(f"Learning rate reduced to {learning_rate}")
    
    # Save final model
    save_model(model, "models/final_model.pth", num_epochs, val_acc, img_size)
    
    # Plot training history
    plot_training_history(
        train_losses, train_accs, val_losses, val_accs,
        "models/training_history.png"
    )
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

