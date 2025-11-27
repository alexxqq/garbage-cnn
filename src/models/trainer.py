"""Training pipeline with MLflow integration"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import mlflow
import mlflow.pytorch
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer with MLflow integration"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training config
        train_cfg = config.training
        self.epochs = train_cfg['epochs']
        self.lr = train_cfg['learning_rate']
        self.save_dir = Path(train_cfg['save_dir'])
        self.save_dir.mkdir(exist_ok=True)
        
        # Setup optimizer
        optimizer_name = train_cfg['optimizer'].lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Setup scheduler
        scheduler_cfg = train_cfg['scheduler']
        if scheduler_cfg['type'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_cfg['step_size'],
                gamma=scheduler_cfg['gamma']
            )
        else:
            self.scheduler = None
        
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.0
        
        # MLflow
        self.mlflow_enabled = config.mlflow['enabled']
        if self.mlflow_enabled:
            mlflow.set_tracking_uri(config.mlflow['tracking_uri'])
            mlflow.set_experiment(config.mlflow['experiment_name'])
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in tqdm(self.train_loader, desc="Training"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        return {
            'loss': train_loss / len(self.train_loader),
            'accuracy': train_correct / train_total
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Validating"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        
        return {
            'loss': val_loss / len(self.val_loader),
            'accuracy': val_correct / val_total
        }
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Learning rate: {self.lr}")
        
        if self.mlflow_enabled:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params({
                    'epochs': self.epochs,
                    'learning_rate': self.lr,
                    'batch_size': self.train_loader.batch_size,
                    'optimizer': self.config.training['optimizer'],
                    'architecture': self.config.model['architecture'],
                    'image_size': self.config.training['image_size']
                })
                
                # Log model architecture
                mlflow.log_text(
                    str(self.model),
                    "model_architecture.txt"
                )
                
                for epoch in range(1, self.epochs + 1):
                    # Train
                    train_metrics = self.train_epoch()
                    
                    # Validate
                    val_metrics = self.validate()
                    
                    # Log metrics
                    mlflow.log_metrics({
                        'train_loss': train_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy'],
                        'epoch': epoch
                    }, step=epoch)
                    
                    # Print progress
                    logger.info(
                        f"Epoch {epoch}/{self.epochs} | "
                        f"Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']*100:.2f}% | "
                        f"Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']*100:.2f}%"
                    )
                    
                    # Save best model
                    if val_metrics['accuracy'] > self.best_val_acc:
                        self.best_val_acc = val_metrics['accuracy']
                        model_path = self.save_dir / f"best_{self.config.model['architecture']}.pth"
                        torch.save(self.model.state_dict(), model_path)
                        logger.info(f"✓ Saved best model (val_acc: {val_metrics['accuracy']*100:.2f}%)")
                        
                        # Log model to MLflow with input example
                        if self.mlflow_enabled and self.config.mlflow['log_model']:
                            # Get a sample input for model signature
                            sample_input = next(iter(self.val_loader))[0][:1].to(self.device)
                            mlflow.pytorch.log_model(
                                self.model,
                                artifact_path="model",
                                input_example=sample_input.cpu().numpy()
                            )
                    
                    # Scheduler step
                    if self.scheduler:
                        self.scheduler.step()
                        mlflow.log_metric('learning_rate', self.scheduler.get_last_lr()[0], step=epoch)
                
                logger.info(f"Training complete! Best validation accuracy: {self.best_val_acc*100:.2f}%")
        else:
            # Training without MLflow
            for epoch in range(1, self.epochs + 1):
                train_metrics = self.train_epoch()
                val_metrics = self.validate()
                
                logger.info(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']*100:.2f}% | "
                    f"Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']*100:.2f}%"
                )
                
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    model_path = self.save_dir / f"best_{self.config.model['architecture']}.pth"
                    torch.save(self.model.state_dict(), model_path)
                    logger.info(f"✓ Saved best model")
                
                if self.scheduler:
                    self.scheduler.step()

