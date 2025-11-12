# Low-Level CNN for Garbage Classification

A low-level implementation of a Convolutional Neural Network (CNN) for garbage type detection, built from scratch using PyTorch tensors. This project demonstrates how CNNs work at a fundamental level by implementing convolution, pooling, and fully connected layers manually.

## Features

- **Low-level CNN implementation**: All layers (convolution, pooling, fully connected) implemented from scratch
- **Manual backpropagation**: Gradient computation and weight updates done manually
- **Garbage classification**: Detects 6 types of garbage: cardboard, glass, metal, paper, plastic, and trash
- **Dataset handling**: Automatic download and organization of TrashNet dataset

## Dataset

The project uses the **TrashNet** dataset, which contains images of garbage items classified into 6 categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The dataset is automatically downloaded and organized into train/validation/test splits when you run the download script.

## Installation

1. Clone the repository or ensure you're in the project directory.

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download and Prepare Dataset

First, download and organize the dataset:
```bash
python download_dataset.py
```

This will:
- Download the TrashNet dataset from GitHub
- Extract it to `data/dataset-resized/`
- Organize it into train/val/test splits in `data/processed/`

### 2. Train the Model

Train the CNN model:
```bash
python train.py
```

Training parameters (can be modified in `train.py`):
- Number of epochs: 20
- Learning rate: 0.001 (with decay every 5 epochs)
- Batch size: 32
- Image size: 64x64

The trained model will be saved to `models/best_model.pth` and `models/final_model.pth`.

### 3. Run Inference

Classify a garbage image:
```bash
python inference.py --image path/to/image.jpg
```

You can also specify a different model:
```bash
python inference.py --image path/to/image.jpg --model models/final_model.pth
```

## Project Structure

```
photoai/
├── cnn_lowlevel.py      # Low-level CNN implementation (Conv2d, MaxPool2d, Linear, etc.)
├── data_loader.py       # Dataset loading and preprocessing
├── download_dataset.py  # Dataset download and organization
├── train.py             # Training script
├── inference.py         # Inference script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── data/               # Dataset directory (created after download)
│   ├── dataset-resized/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
└── models/             # Trained models (created after training)
    ├── best_model.pth
    ├── final_model.pth
    └── training_history.png
```

## CNN Architecture

The network consists of:

1. **Convolutional Block 1**:
   - Conv2d(3, 32, 3x3) + ReLU
   - Conv2d(32, 32, 3x3) + ReLU
   - MaxPool2d(2x2)

2. **Convolutional Block 2**:
   - Conv2d(32, 64, 3x3) + ReLU
   - Conv2d(64, 64, 3x3) + ReLU
   - MaxPool2d(2x2)

3. **Convolutional Block 3**:
   - Conv2d(64, 128, 3x3) + ReLU
   - MaxPool2d(2x2)

4. **Fully Connected Layers**:
   - Linear(128*8*8, 512) + ReLU + Dropout(0.5)
   - Linear(512, 6)

## Low-Level Implementation Details

All layers are implemented from scratch:

- **Conv2d**: Manual convolution with nested loops, weight initialization using He initialization
- **MaxPool2d**: Manual max pooling with index tracking for backpropagation
- **Linear**: Matrix multiplication for fully connected layers
- **ReLU**: Element-wise activation with gradient tracking
- **Dropout**: Regularization during training
- **Backpropagation**: Manual gradient computation through all layers

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pillow
- Matplotlib
- tqdm
- scikit-learn

## Notes

- The low-level implementation is educational and demonstrates CNN fundamentals
- For production use, consider using PyTorch's built-in layers for better performance
- Training may take some time depending on your hardware
- GPU is recommended but not required (CPU training will be slower)

## License

This project is for educational purposes. The TrashNet dataset is available from the original repository: https://github.com/garythung/trashnet

