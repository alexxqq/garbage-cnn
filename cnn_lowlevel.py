"""
Low-level CNN Implementation for Garbage Classification
Implements CNN layers from scratch using PyTorch tensors
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

class Conv2d:
    """Low-level 2D Convolution Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights using He initialization
        k = np.sqrt(1.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = torch.randn(out_channels, in_channels, 
                                  self.kernel_size[0], self.kernel_size[1]) * k
        self.bias = torch.zeros(out_channels)
        
        # Gradients
        self.weight_grad = None
        self.bias_grad = None
        self.input = None
    
    def forward(self, x):
        """Forward pass: manual convolution"""
        self.input = x.clone()
        batch_size, in_channels, height, width = x.shape
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Add padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[1], self.padding[1], 
                         self.padding[0], self.padding[0]), mode='constant', value=0)
        
        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, out_height, out_width)
        
        # Manual convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride[0]
                        w_start = ow * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        # Extract patch and compute dot product
                        patch = x[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, oh, ow] = torch.sum(patch * self.weight[oc]) + self.bias[oc]
        
        return output
    
    def backward(self, grad_output):
        """Backward pass: compute gradients"""
        batch_size, in_channels, height, width = self.input.shape
        _, out_channels, out_height, out_width = grad_output.shape
        
        # Initialize gradients
        grad_input_padded = torch.zeros(
            batch_size, in_channels, 
            height + 2 * self.padding[0], 
            width + 2 * self.padding[1]
        )
        self.weight_grad = torch.zeros_like(self.weight)
        self.bias_grad = torch.zeros_like(self.bias)
        
        # Pad input for gradient computation
        padded_input = F.pad(self.input, (self.padding[1], self.padding[1], 
                                         self.padding[0], self.padding[0]), 
                           mode='constant', value=0)
        
        # Compute gradients
        for b in range(batch_size):
            for oc in range(out_channels):
                self.bias_grad[oc] += torch.sum(grad_output[b, oc])
                
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride[0]
                        w_start = ow * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        grad_val = grad_output[b, oc, oh, ow]
                        
                        # Weight gradients
                        self.weight_grad[oc] += padded_input[b, :, h_start:h_end, w_start:w_end] * grad_val
                        
                        # Input gradients (on padded input)
                        grad_input_padded[b, :, h_start:h_end, w_start:w_end] += self.weight[oc] * grad_val
        
        # Normalize gradients by batch size
        self.weight_grad /= batch_size
        self.bias_grad /= batch_size
        
        # Remove padding from input gradients
        if self.padding[0] > 0 or self.padding[1] > 0:
            h_start = self.padding[0]
            h_end = -self.padding[0] if self.padding[0] > 0 else None
            w_start = self.padding[1]
            w_end = -self.padding[1] if self.padding[1] > 0 else None
            grad_input = grad_input_padded[:, :, h_start:h_end, w_start:w_end]
        else:
            grad_input = grad_input_padded
        
        return grad_input
    
    def update(self, learning_rate):
        """Update weights using gradient descent"""
        if self.weight_grad is not None:
            self.weight -= learning_rate * self.weight_grad
            self.bias -= learning_rate * self.bias_grad


class MaxPool2d:
    """Low-level 2D Max Pooling Layer"""
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.input = None
        self.max_indices = None
    
    def forward(self, x):
        """Forward pass: manual max pooling"""
        self.input = x.clone()
        batch_size, channels, height, width = x.shape
        
        out_height = (height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width - self.kernel_size[1]) // self.stride[1] + 1
        
        output = torch.zeros(batch_size, channels, out_height, out_width)
        self.max_indices = {}
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride[0]
                        w_start = ow * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        patch = x[b, c, h_start:h_end, w_start:w_end]
                        patch_flat = patch.reshape(-1)
                        max_val, max_idx = torch.max(patch_flat, 0)
                        output[b, c, oh, ow] = max_val
                        self.max_indices[(b, c, oh, ow)] = (
                            h_start + max_idx.item() // self.kernel_size[1],
                            w_start + max_idx.item() % self.kernel_size[1]
                        )
        
        return output
    
    def backward(self, grad_output):
        """Backward pass: only propagate to max positions"""
        grad_input = torch.zeros_like(self.input)
        
        for (b, c, oh, ow), (h_idx, w_idx) in self.max_indices.items():
            grad_input[b, c, h_idx, w_idx] += grad_output[b, c, oh, ow]
        
        return grad_input


class Linear:
    """Low-level Fully Connected (Linear) Layer"""
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier initialization
        k = np.sqrt(1.0 / in_features)
        self.weight = torch.randn(out_features, in_features) * k
        self.bias = torch.zeros(out_features)
        
        self.weight_grad = None
        self.bias_grad = None
        self.input = None
    
    def forward(self, x):
        """Forward pass: matrix multiplication"""
        self.input = x.clone()
        # x: [batch_size, in_features]
        # output: [batch_size, out_features]
        return x @ self.weight.t() + self.bias
    
    def backward(self, grad_output):
        """Backward pass: compute gradients"""
        # grad_output: [batch_size, out_features]
        batch_size = grad_output.shape[0]
        
        # Weight gradients
        self.weight_grad = grad_output.t() @ self.input / batch_size
        
        # Bias gradients
        self.bias_grad = torch.mean(grad_output, dim=0)
        
        # Input gradients
        grad_input = grad_output @ self.weight
        
        return grad_input
    
    def update(self, learning_rate):
        """Update weights using gradient descent"""
        if self.weight_grad is not None:
            self.weight -= learning_rate * self.weight_grad
            self.bias -= learning_rate * self.bias_grad


class ReLU:
    """ReLU Activation Function"""
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        """Forward pass"""
        self.input = x.clone()
        return torch.clamp(x, min=0)
    
    def backward(self, grad_output):
        """Backward pass: zero gradient for negative inputs"""
        return grad_output * (self.input > 0).float()


class Dropout:
    """Dropout Layer for Regularization"""
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None
    
    def forward(self, x):
        """Forward pass"""
        if self.training:
            self.mask = (torch.rand_like(x) > self.p).float()
            return x * self.mask / (1 - self.p)
        return x
    
    def backward(self, grad_output):
        """Backward pass"""
        if self.training:
            return grad_output * self.mask / (1 - self.p)
        return grad_output


class GarbageCNN:
    """Low-level CNN for Garbage Classification"""
    def __init__(self, num_classes=6, img_size=64):
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Conv layers
        self.conv1 = Conv2d(3, 32, 3, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2d(32, 32, 3, padding=1)
        self.relu2 = ReLU()
        self.pool1 = MaxPool2d(2, 2)
        
        self.conv3 = Conv2d(32, 64, 3, padding=1)
        self.relu3 = ReLU()
        self.conv4 = Conv2d(64, 64, 3, padding=1)
        self.relu4 = ReLU()
        self.pool2 = MaxPool2d(2, 2)
        
        self.conv5 = Conv2d(64, 128, 3, padding=1)
        self.relu5 = ReLU()
        self.pool3 = MaxPool2d(2, 2)
        
        # Calculate feature map size after all conv/pool layers
        # Input: img_size x img_size
        # After pool1 (stride 2): img_size/2 x img_size/2
        # After pool2 (stride 2): img_size/4 x img_size/4
        # After pool3 (stride 2): img_size/8 x img_size/8
        feature_size = (img_size // 8) * (img_size // 8) * 128
        self.feature_size = feature_size
        
        # Fully connected layers
        self.fc1 = Linear(feature_size, 512)
        self.relu6 = ReLU()
        self.dropout = Dropout(0.5)
        self.fc2 = Linear(512, num_classes)
        
        self.layers = [
            self.conv1, self.relu1, self.conv2, self.relu2, self.pool1,
            self.conv3, self.relu3, self.conv4, self.relu4, self.pool2,
            self.conv5, self.relu5, self.pool3,
            self.fc1, self.relu6, self.dropout, self.fc2
        ]
    
    def forward(self, x):
        """Forward pass through the network"""
        # x: [batch_size, 3, 64, 64]
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.conv4.forward(x)
        x = self.relu4.forward(x)
        x = self.pool2.forward(x)
        
        x = self.conv5.forward(x)
        x = self.relu5.forward(x)
        x = self.pool3.forward(x)
        
        # Flatten
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        x = self.fc1.forward(x)
        x = self.relu6.forward(x)
        x = self.dropout.forward(x)
        x = self.fc2.forward(x)
        
        return x
    
    def backward(self, grad_output):
        """Backward pass through the network"""
        grad = grad_output
        
        # Backward through layers in reverse
        grad = self.fc2.backward(grad)
        grad = self.dropout.backward(grad)
        grad = self.relu6.backward(grad)
        grad = self.fc1.backward(grad)
        
        # Reshape for conv layers
        batch_size = grad.shape[0]
        feature_h = self.img_size // 8
        feature_w = self.img_size // 8
        grad = grad.view(batch_size, 128, feature_h, feature_w)
        
        grad = self.pool3.backward(grad)
        grad = self.relu5.backward(grad)
        grad = self.conv5.backward(grad)
        
        grad = self.pool2.backward(grad)
        grad = self.relu4.backward(grad)
        grad = self.conv4.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.conv3.backward(grad)
        
        grad = self.pool1.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)
        
        return grad
    
    def update(self, learning_rate):
        """Update all layer weights"""
        self.conv1.update(learning_rate)
        self.conv2.update(learning_rate)
        self.conv3.update(learning_rate)
        self.conv4.update(learning_rate)
        self.conv5.update(learning_rate)
        self.fc1.update(learning_rate)
        self.fc2.update(learning_rate)
    
    def set_training(self, training=True):
        """Set training/eval mode"""
        self.dropout.training = training
    
    def zero_grad(self):
        """Reset gradients (handled in backward)"""
        pass
    
    def to(self, device):
        """Move all parameters to the specified device"""
        # Move Conv2d layers
        self.conv1.weight = self.conv1.weight.to(device)
        self.conv1.bias = self.conv1.bias.to(device)
        self.conv2.weight = self.conv2.weight.to(device)
        self.conv2.bias = self.conv2.bias.to(device)
        self.conv3.weight = self.conv3.weight.to(device)
        self.conv3.bias = self.conv3.bias.to(device)
        self.conv4.weight = self.conv4.weight.to(device)
        self.conv4.bias = self.conv4.bias.to(device)
        self.conv5.weight = self.conv5.weight.to(device)
        self.conv5.bias = self.conv5.bias.to(device)
        
        # Move Linear layers
        self.fc1.weight = self.fc1.weight.to(device)
        self.fc1.bias = self.fc1.bias.to(device)
        self.fc2.weight = self.fc2.weight.to(device)
        self.fc2.bias = self.fc2.bias.to(device)
        
        return self


def cross_entropy_loss(predictions, targets):
    """Low-level cross-entropy loss"""
    batch_size = predictions.shape[0]
    
    # Softmax
    exp_pred = torch.exp(predictions - torch.max(predictions, dim=1, keepdim=True)[0])
    softmax_pred = exp_pred / torch.sum(exp_pred, dim=1, keepdim=True)
    
    # One-hot encode targets
    targets_one_hot = torch.zeros_like(softmax_pred)
    targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
    
    # Cross-entropy
    loss = -torch.sum(targets_one_hot * torch.log(softmax_pred + 1e-10)) / batch_size
    
    # Gradient of loss w.r.t. predictions
    grad = (softmax_pred - targets_one_hot) / batch_size
    
    return loss, grad

