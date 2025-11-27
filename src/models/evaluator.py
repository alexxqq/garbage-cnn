"""Model evaluation and metrics"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluate model and generate metrics"""
    
    def __init__(self, model, data_loader, device, class_names):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.class_names = class_names
    
    def evaluate(self, save_path: str = "models/metrics"):
        """Run evaluation and generate metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        logger.info("Evaluating model...")
        with torch.no_grad():
            for imgs, labels in self.data_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)
        
        # Save metrics
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        self._plot_confusion_matrix(cm, save_path / "confusion_matrix.png")
        self._plot_confusion_matrix_normalized(cm, save_path / "confusion_matrix_normalized.png")
        
        # Per-class accuracy
        self._plot_per_class_accuracy(all_labels, all_preds, save_path / "per_class_accuracy.png")
        
        # Precision/Recall/F1
        self._plot_metrics_comparison(report, save_path / "precision_recall_f1.png")
        
        # Save text report
        self._save_text_report(accuracy, report, save_path / "metrics.txt")
        
        logger.info(f"Metrics saved to {save_path}")
        logger.info(f"Overall Accuracy: {accuracy*100:.2f}%")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def _plot_confusion_matrix(self, cm, save_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix_normalized(self, cm, save_path):
        """Plot normalized confusion matrix"""
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_accuracy(self, all_labels, all_preds, save_path):
        """Plot per-class accuracy"""
        per_class_acc = []
        for i, cls in enumerate(self.class_names):
            class_mask = all_labels == i
            if class_mask.sum() > 0:
                class_acc = (all_preds[class_mask] == all_labels[class_mask]).mean()
                per_class_acc.append(class_acc * 100)
            else:
                per_class_acc.append(0)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.class_names, per_class_acc, color='steelblue', edgecolor='black')
        plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.ylim(0, 105)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, per_class_acc):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, report, save_path):
        """Plot precision, recall, F1 comparison"""
        precisions = [report[cls]['precision'] for cls in self.class_names]
        recalls = [report[cls]['recall'] for cls in self.class_names]
        f1_scores = [report[cls]['f1-score'] for cls in self.class_names]
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precisions, width, label='Precision', color='#2ecc71', edgecolor='black')
        plt.bar(x, recalls, width, label='Recall', color='#3498db', edgecolor='black')
        plt.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c', edgecolor='black')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Precision, Recall, and F1-Score by Class', fontsize=16, fontweight='bold')
        plt.xticks(x, self.class_names, rotation=45, ha='right')
        plt.legend(fontsize=11)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_text_report(self, accuracy, report, save_path):
        """Save text report"""
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL EVALUATION METRICS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write("Classification Report:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-" * 70 + "\n")
            for cls in self.class_names:
                f.write(f"{cls:<15} {report[cls]['precision']:.3f}        "
                       f"{report[cls]['recall']:.3f}        {report[cls]['f1-score']:.3f}        "
                       f"{int(report[cls]['support']):<10}\n")
            f.write(f"\nMacro Avg: {report['macro avg']['precision']:.3f}        "
                   f"{report['macro avg']['recall']:.3f}        {report['macro avg']['f1-score']:.3f}\n")

