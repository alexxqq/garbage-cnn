"""
High-Level CNN Inference for Garbage Classification
Uses ResNet50 (Transfer Learning)
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import argparse
from pathlib import Path

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def create_model(num_classes=6):
    """Create ResNet50 model"""
    model = models.resnet50(weights=None)  # No pre-trained weights for inference
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def predict_image(model, image_path, device, img_size=224):
    """Predict garbage type for a single image"""
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return predicted_class.item(), confidence.item(), probabilities[0].cpu()

def main():
    parser = argparse.ArgumentParser(description='Garbage Classification Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/best_resnet50.pth', help='Path to model file')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (default: 224 for ResNet)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not Path(args.model).exists():
        print(f"Model file {args.model} not found!")
        print("Please train the model first using: python train.py")
        return
    
    # Load model
    model = create_model(num_classes=len(CLASSES)).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded ResNet50 from {args.model}")
    
    if not Path(args.image).exists():
        print(f"Image file {args.image} not found!")
        return
    
    # Predict
    predicted_class, confidence, probabilities = predict_image(
        model, args.image, device, args.img_size
    )
    
    # Print results
    print(f"\nImage: {args.image}")
    print(f"Predicted Class: {CLASSES[predicted_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nAll Class Probabilities:")
    for i, (cls, prob) in enumerate(zip(CLASSES, probabilities)):
        marker = " <--" if i == predicted_class else ""
        print(f"  {cls}: {prob * 100:.2f}%{marker}")

if __name__ == "__main__":
    main()

