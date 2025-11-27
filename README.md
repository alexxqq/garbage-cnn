# Garbage Classification

Production-ready ResNet50 transfer learning model for garbage classification with MLflow tracking, FastAPI, and Streamlit UI.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Prepare data (downloads TrashNet + Kaggle)
python prepare_data.py

# Train
python train.py

# Evaluate
python evaluate.py

# Test on Kaggle dataset
python test_kaggle.py

# View MLflow
mlflow ui

# Run API
python api/main.py

# Run UI
streamlit run ui/app.py
```

## Classes

cardboard, glass, metal, paper, plastic, trash

