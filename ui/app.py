"""
Streamlit UI for Garbage Classification
"""
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config
from src.models.model_factory import create_model

# Page config
st.set_page_config(
    page_title="Garbage Classification",
    page_icon="🗑️",
    layout="wide"
)

# Load config
@st.cache_resource
def load_config():
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "config.yaml"
    return Config(str(config_path))

@st.cache_resource
def load_model():
    """Load model with caching"""
    config = load_config()
    project_root = Path(__file__).parent.parent
    model_path = project_root / config.ui['model_path']
    
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        architecture=config.model['architecture'],
        num_classes=len(config.model['classes']),
        pretrained=False
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device, config

# Main UI
st.title("🗑️ Garbage Classification System")
st.markdown("---")

config = load_config()
model_data = load_model()

if model_data is None:
    st.stop()

model, device, config = model_data
classes = config.model['classes']

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose page", ["Predict", "Model Info", "MLflow Metrics"])

if page == "Predict":
    st.header("Image Classification")
    
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of garbage to classify"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
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
            
            # Display results
            st.subheader("Prediction Results")
            st.metric(
                "Predicted Class",
                classes[pred_idx].upper(),
                f"{confidence*100:.2f}% confidence"
            )
            
            # Probabilities chart
            prob_dict = {classes[i]: float(probs[0][i].item()) for i in range(len(classes))}
            df = pd.DataFrame(list(prob_dict.items()), columns=['Class', 'Probability'])
            df = df.sort_values('Probability', ascending=True)
            
            fig = px.bar(
                df,
                x='Probability',
                y='Class',
                orientation='h',
                title="Class Probabilities",
                color='Probability',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Model Info":
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        st.json({
            "Architecture": config.model['architecture'],
            "Classes": config.model['classes'],
            "Image Size": config.training['image_size'],
            "Batch Size": config.training['batch_size'],
            "Epochs": config.training['epochs'],
            "Learning Rate": config.training['learning_rate']
        })
    
    with col2:
        st.subheader("Model Statistics")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        st.metric("Total Parameters", f"{total_params:,}")
        st.metric("Trainable Parameters", f"{trainable_params:,}")
        st.metric("Device", str(device))

elif page == "MLflow Metrics":
    st.header("MLflow Experiment Tracking")
    
    mlruns_dir = Path("mlruns")
    if not mlruns_dir.exists():
        st.warning("MLflow runs not found. Run training with MLflow enabled to see metrics.")
    else:
        st.info("MLflow UI available at: http://localhost:5000")
        st.code("mlflow ui", language="bash")
        
        # Try to load latest run
        try:
            import mlflow
            mlflow.set_tracking_uri("file:./mlruns")
            experiment = mlflow.get_experiment_by_name(config.mlflow['experiment_name'])
            
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=1)
                if not runs.empty:
                    st.subheader("Latest Run Metrics")
                    latest_run = runs.iloc[0]
                    
                    metrics_cols = [col for col in latest_run.index if col.startswith('metrics.')]
                    if metrics_cols:
                        metrics_df = pd.DataFrame({
                            'Metric': [col.replace('metrics.', '') for col in metrics_cols],
                            'Value': [latest_run[col] for col in metrics_cols]
                        })
                        st.dataframe(metrics_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading MLflow data: {e}")

