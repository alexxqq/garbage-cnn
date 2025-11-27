"""
FastAPI REST API for Garbage Classification
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.models.model_factory import create_model

# Setup - config path relative to project root
import os
project_root = Path(__file__).parent.parent
config_path = project_root / "configs" / "config.yaml"
config = Config(str(config_path))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Garbage Classification API",
    description="REST API for garbage classification using ResNet",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = config.model['classes']

def load_model():
    """Load trained model"""
    global model
    model_path = Path(config.api['model_path'])
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = create_model(
        architecture=config.model['architecture'],
        num_classes=len(classes),
        pretrained=False
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "message": "Garbage Classification API",
        "classes": classes
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict garbage class from uploaded image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform
        transform = transforms.Compose([
            transforms.Resize((config.training['image_size'], config.training['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = outputs.argmax(1).item()
            confidence = probs[0][pred_idx].item()
        
        # Format response
        predictions = {
            classes[i]: float(probs[0][i].item())
            for i in range(len(classes))
        }
        
        return JSONResponse({
            "predicted_class": classes[pred_idx],
            "confidence": float(confidence),
            "all_predictions": predictions
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict multiple images"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    transform = transforms.Compose([
        transforms.Resize((config.training['image_size'], config.training['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = outputs.argmax(1).item()
                confidence = probs[0][pred_idx].item()
            
            results.append({
                "filename": file.filename,
                "predicted_class": classes[pred_idx],
                "confidence": float(confidence)
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({"results": results})

if __name__ == "__main__":
    import uvicorn
    import os
    os.chdir(Path(__file__).parent.parent)  # Change to project root
    uvicorn.run(
        "api.main:app",
        host=config.api['host'],
        port=config.api['port'],
        reload=True
    )

