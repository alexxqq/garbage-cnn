"""
Inference Script for Garbage Classification
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
from pathlib import Path
from cnn_lowlevel import GarbageCNN
from train import load_model

def predict_image(model, image_path, device, img_size=64):
    """Predict garbage type for a single image"""
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Set model to eval mode
    model.set_training(False)
    
    # Forward pass
    with torch.no_grad():
        predictions = model.forward(image_tensor)
        probabilities = torch.softmax(predictions, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return predicted_class.item(), confidence.item(), probabilities[0].cpu()


def main():
    parser = argparse.ArgumentParser(description='Garbage Classification Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Path to model file')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    
    args = parser.parse_args()
    
    # Classes
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    if not Path(args.model).exists():
        print(f"Model file {args.model} not found!")
        print("Please train the model first using: python train.py")
        return
    
    # Load checkpoint to get img_size
    checkpoint = torch.load(args.model, map_location=device)
    saved_img_size = checkpoint.get('img_size', args.img_size)
    
    # Use saved img_size if available, otherwise use command line argument
    model_img_size = saved_img_size if saved_img_size else args.img_size
    
    if saved_img_size and saved_img_size != args.img_size:
        print(f"Warning: Model was trained with img_size={saved_img_size}, but {args.img_size} was specified.")
        print(f"Using img_size={saved_img_size} from checkpoint.")
    
    model = GarbageCNN(num_classes=len(classes), img_size=model_img_size)
    model = model.to(device)
    
    try:
        load_model(model, args.model, device)
        print(f"Loaded model from {args.model} (img_size={model_img_size})")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Image file {args.image} not found!")
        return
    
    # Predict
    predicted_class, confidence, probabilities = predict_image(
        model, args.image, device, model_img_size
    )
    
    # Print results
    print(f"\nImage: {args.image}")
    print(f"\nPredicted Class: {classes[predicted_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nAll Class Probabilities:")
    for i, (cls, prob) in enumerate(zip(classes, probabilities)):
        marker = " <--" if i == predicted_class else ""
        print(f"  {cls}: {prob * 100:.2f}%{marker}")


if __name__ == "__main__":
    main()

