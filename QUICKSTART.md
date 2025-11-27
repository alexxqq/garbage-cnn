# Quick Start Guide

## 🚀 Complete Workflow

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data (Downloads TrashNet + Kaggle)

```bash
python prepare_data.py
```

This will:
- Download TrashNet dataset from GitHub
- Download Kaggle dataset (`asdasdasasdas/garbage-classification`)
- Organize into `data/processed/train/`, `data/processed/val/`, `data/processed/test/`
- Create 70/15/15 split

### Step 3: Train Model

```bash
python train.py
```

This will:
- Load prepared data
- Create ResNet50 model (or configured architecture)
- Train with MLflow tracking
- Save best model to `models/best_resnet50.pth`
- Log all metrics to MLflow

### Step 4: View Training Metrics (MLflow UI)

```bash
mlflow ui
```

Then open: http://localhost:5000

You'll see:
- All training runs
- Metrics (loss, accuracy) over epochs
- Parameters (learning rate, batch size, etc.)
- Model artifacts

### Step 5: Evaluate Model

```bash
python evaluate.py
```

Generates:
- `models/metrics/confusion_matrix.png`
- `models/metrics/per_class_accuracy.png`
- `models/metrics/precision_recall_f1.png`
- `models/metrics/metrics.txt`

### Step 6: Run API Server

```bash
./run_api.sh
# OR
python api/main.py
```

API available at: http://localhost:8000

**Test it:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/test/plastic1.png"
```

**API Docs:** http://localhost:8000/docs

### Step 7: Run Streamlit UI

```bash
./run_ui.sh
# OR
streamlit run ui/app.py
```

UI available at: http://localhost:8501

Features:
- **Predict**: Upload images and get predictions
- **Model Info**: View model configuration
- **MLflow Metrics**: View experiment tracking

## 📁 Project Structure

```
photoai/
├── src/                    # Source code
│   ├── data/              # Dataset loading
│   ├── models/            # Model definitions & training
│   └── utils/             # Utilities
├── configs/               # Configuration files
│   └── config.yaml       # Main config
├── api/                   # FastAPI REST API
├── ui/                    # Streamlit UI
├── data/                  # Data storage
│   ├── raw/              # Raw datasets
│   └── processed/        # Processed splits
├── models/                # Trained models
├── mlruns/               # MLflow runs
└── logs/                 # Log files
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

- **Model Architecture**: `resnet18`, `resnet50`, `resnet101`, `efficientnet_b0`, `mobilenet_v3`
- **Training**: epochs, batch_size, learning_rate
- **Datasets**: Enable/disable TrashNet or Kaggle
- **MLflow**: Tracking URI, experiment name
- **API/UI**: Ports, model paths

## 🔍 Testing External Datasets

Use the existing `test_external_dataset.py`:

```bash
# With labels (class folders)
python test_external_dataset.py --dataset /path/to/dataset --has-labels

# Without labels (just predictions)
python test_external_dataset.py --dataset /path/to/dataset
```

## 📊 MLflow Features

- **Automatic Logging**: All training runs logged
- **Parameter Tracking**: Hyperparameters, configs
- **Metric Tracking**: Loss, accuracy per epoch
- **Model Versioning**: Track model checkpoints
- **UI**: Visualize experiments, compare runs

## 🎯 Next Steps

1. **Experiment**: Try different architectures in `config.yaml`
2. **Tune Hyperparameters**: Adjust learning rate, batch size
3. **Add Datasets**: Add more datasets in config
4. **Deploy**: Use API for production deployment
5. **Monitor**: Use MLflow to track model performance

## 🐛 Troubleshooting

**Kaggle dataset not downloading?**
- Make sure `kagglehub` is installed: `pip install kagglehub`
- Check Kaggle credentials if needed

**MLflow UI not showing runs?**
- Make sure `mlflow.enabled: true` in config
- Check `mlruns/` directory exists

**Model not found?**
- Run training first: `python train.py`
- Check `models/` directory for saved models

**Import errors?**
- Make sure you're in project root
- Install all dependencies: `pip install -r requirements.txt`

