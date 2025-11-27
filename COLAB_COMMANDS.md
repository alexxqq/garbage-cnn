# Colab Commands - Quick Reference

## 📋 Setup (Run Once)

```bash
# Clone your repo
!git clone https://github.com/your-username/photoai.git
# OR upload your project files manually

# Navigate to project
%cd photoai

# Install dependencies
!pip install -r requirements.txt
```

## 🗂️ Prepare Data

```bash
# Download and organize datasets (TrashNet + Kaggle)
!python prepare_data.py
```

## 🚂 Train Model

```bash
# Train with MLflow tracking
!python train.py
```

**What this does:**
- Downloads datasets (if not already done)
- Trains ResNet50 model
- Logs everything to MLflow
- Saves best model to `models/best_resnet50.pth`

## 📊 Evaluate Model

```bash
# Evaluate on test set and generate metrics
!python evaluate.py
```

**Output:**
- `models/metrics/confusion_matrix.png`
- `models/metrics/per_class_accuracy.png`
- `models/metrics/metrics.txt`

## 📥 Download Results

```bash
# Download mlruns (MLflow experiments)
!zip -r mlruns.zip mlruns/
from google.colab import files
files.download('mlruns.zip')

# Download trained model
files.download('models/best_resnet50.pth')

# Download metrics
!zip -r metrics.zip models/metrics/
files.download('metrics.zip')
```

## 🔍 View MLflow in Colab

```bash
# Start MLflow UI (runs in background)
!mlflow ui --host 0.0.0.0 --port 5000 &
```

Then use Colab's `ngrok` or port forwarding to access it.

## 📝 Complete Workflow (Copy-Paste)

```python
# === SETUP ===
!git clone https://github.com/your-username/photoai.git
%cd photoai
!pip install -r requirements.txt

# === PREPARE DATA ===
!python prepare_data.py

# === TRAIN ===
!python train.py

# === EVALUATE ===
!python evaluate.py

# === DOWNLOAD RESULTS ===
!zip -r mlruns.zip mlruns/
!zip -r results.zip models/
from google.colab import files
files.download('mlruns.zip')
files.download('results.zip')
```

## 🎯 Quick Commands Summary

| Task | Command |
|------|---------|
| **Setup** | `!pip install -r requirements.txt` |
| **Prepare Data** | `!python prepare_data.py` |
| **Train** | `!python train.py` |
| **Evaluate** | `!python evaluate.py` |
| **Download MLflow** | `!zip -r mlruns.zip mlruns/` then `files.download('mlruns.zip')` |
| **Download Model** | `files.download('models/best_resnet50.pth')` |

## 💡 Tips

1. **Check GPU**: `!nvidia-smi` (Colab usually has free GPU)
2. **Monitor Training**: Watch the output for progress
3. **Save Checkpoints**: Models auto-save to `models/` folder
4. **View Logs**: Check `logs/training.log` if needed

## 🔄 After Downloading to Local

```bash
# Extract mlruns
unzip mlruns.zip

# View in MLflow UI
mlflow ui
# Open http://localhost:5000
```

