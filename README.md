# Garbage Classification - Production Ready Project

A production-ready deep learning project for garbage classification using transfer learning with ResNet. Features MLflow experiment tracking, FastAPI REST API, Streamlit UI, and support for multiple datasets.

## 🏗️ Project Structure

```
photoai/
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset classes
│   │   └── dataset_loader.py   # Dataset download and preparation
│   ├── models/
│   │   ├── model_factory.py    # Model creation
│   │   ├── trainer.py          # Training with MLflow
│   │   └── evaluator.py        # Model evaluation
│   └── utils/
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging utilities
├── configs/
│   └── config.yaml             # Project configuration
├── api/
│   └── main.py                 # FastAPI REST API
├── ui/
│   └── app.py                  # Streamlit UI
├── train.py                    # Main training script
├── prepare_data.py             # Data preparation script
└── requirements.txt            # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Download and organize datasets (TrashNet + Kaggle):

```bash
python prepare_data.py
```

### 3. Train Model

Train with MLflow tracking:

```bash
python train.py
```

### 4. View MLflow UI

```bash
mlflow ui
```

Open http://localhost:5000 to see experiment metrics.

### 5. Run API Server

```bash
cd api
python main.py
```

API will be available at http://localhost:8000

### 6. Run Streamlit UI

```bash
streamlit run ui/app.py
```

UI will be available at http://localhost:8501

## 📊 Features

### Data Preparation
- **Multiple Dataset Support**: TrashNet (GitHub) + Kaggle datasets
- **Automatic Organization**: Train/Val/Test splits
- **Flexible Structure**: Handles different folder structures

### Training
- **MLflow Integration**: Automatic experiment tracking
- **Multiple Architectures**: ResNet18, ResNet50, ResNet101, EfficientNet, MobileNet
- **Configurable**: YAML-based configuration
- **Checkpointing**: Automatic best model saving

### API
- **FastAPI REST API**: Production-ready endpoints
- **Single & Batch Prediction**: `/predict` and `/predict/batch`
- **CORS Enabled**: Ready for web integration

### UI
- **Streamlit Interface**: Interactive model testing
- **Real-time Predictions**: Upload and classify images
- **Model Info**: View configuration and statistics
- **MLflow Integration**: View experiment metrics

### Evaluation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualizations**: Confusion matrices, per-class metrics
- **Reports**: Text and visual outputs

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

- **Model**: Architecture, classes, pretrained weights
- **Training**: Epochs, batch size, learning rate, optimizer
- **Data**: Dataset sources, splits
- **MLflow**: Tracking URI, experiment name
- **API/UI**: Ports, model paths

## 📝 Usage Examples

### Training

```bash
# Train with default config
python train.py

# Training automatically:
# - Downloads datasets
# - Organizes into train/val/test
# - Trains with MLflow tracking
# - Saves best model
```

### API Usage

```bash
# Start API
cd api && python main.py

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### UI Usage

```bash
# Start UI
streamlit run ui/app.py

# Navigate to:
# - Predict: Upload and classify images
# - Model Info: View model configuration
# - MLflow Metrics: View experiment tracking
```

## 🔧 Architecture

- **ResNet50** (default): Transfer learning from ImageNet
- **6 Classes**: cardboard, glass, metal, paper, plastic, trash
- **Image Size**: 224x224 (ResNet standard)
- **Data Augmentation**: Random flip, rotation

## 📈 MLflow Tracking

All training runs are automatically logged to MLflow:

- **Parameters**: Architecture, hyperparameters, config
- **Metrics**: Loss, accuracy per epoch
- **Artifacts**: Model checkpoints, metrics plots
- **UI**: Access via `mlflow ui`

## 🎯 Best Practices Implemented

✅ **Modular Architecture**: Separated data, models, utils  
✅ **Configuration Management**: YAML-based configs  
✅ **Experiment Tracking**: MLflow integration  
✅ **Logging**: Structured logging throughout  
✅ **API Design**: RESTful endpoints with FastAPI  
✅ **UI**: Interactive Streamlit interface  
✅ **Error Handling**: Proper exception handling  
✅ **Type Hints**: Type annotations for clarity  
✅ **Documentation**: Docstrings and README  

## 📦 Datasets Supported

1. **TrashNet** (GitHub): Default dataset
2. **Kaggle**: `asdasdasasdas/garbage-classification`

Add more datasets in `configs/config.yaml`

## 🛠️ Development

```bash
# Install in development mode
pip install -e .

# Run tests (when implemented)
pytest tests/
```

## 🚀 Training in Google Colab

See [COLAB_COMMANDS.md](COLAB_COMMANDS.md) for quick command reference.

**Quick start:**
```bash
!git clone https://github.com/your-username/photoai.git
%cd photoai
!pip install -r requirements.txt
!python prepare_data.py
!python train.py
!python evaluate.py
```

## 📄 License

MIT License
