
```bash
pip install -r requirements.txt
```

### 1. Download Dataset

```bash
python download_dataset.py
```

This downloads and organizes the TrashNet dataset into `data/processed/`.

### 2. Train Model

```bash
python train.py
```

### 3. Run Inference

```bash
python inference.py --image path/to/image.jpg
```

Or specify a model:
```bash
python inference.py --image path/to/image.jpg --model models/best_highlevel.pth
```
