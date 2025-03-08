# Dynamic Temperature Goodness for Enhanced Forward-Forward Learning (DTG-FF)

This repository implements the Dynamic Temperature Goodness (DTG) mechanism for the Forward-Forward algorithm as described in the paper "Dynamic Temperature Goodness for Enhanced Forward-Forward Learning". DTG-FF systematically addresses fundamental limitations of the original Forward-Forward algorithm while maintaining its elegance and biological plausibility.

## Overview

The Forward-Forward (FF) algorithm, introduced by Geoffrey Hinton, offers an alternative to backpropagation by using two forward passes on positive and negative data. However, the fixed discrimination criteria in FF limits its performance on complex datasets.

DTG-FF enhances the FF algorithm through:

1. **Dynamic Temperature Mechanism**: Adaptively adjusts discrimination criteria based on feature clarity
2. **Feature-Driven Optimization Framework**: Incorporates temperature adjustment, history-based stabilization, and adaptive margins
3. **Self-Adaptive Thresholds**: Learns optimal discrimination boundaries for each layer

## Key Components

### DTGLayer

The core building block that implements dynamic temperature goodness calculation. Each layer:
- Calculates feature statistics to determine clarity
- Dynamically adjusts temperature based on feature distribution
- Uses normalized L2 distance with temperature scaling for goodness computation
- Maintains adaptive thresholds with softplus activation

### FF_DTG_Model

The complete model architecture that:
- Processes both positive and negative samples
- Calculates losses with dynamic margins and adaptive thresholds
- Provides per-layer accuracy metrics
- Supports classification tasks via feature concatenation

### FF_DTG_Config

Configuration class to manage:
- Dataset settings (MNIST/CIFAR-10)
- Model architecture (layer count, hidden dimensions)
- Training parameters (learning rate, weight decay)

## Performance

DTG-FF achieves:
- 98.7% accuracy on MNIST
- 60.1% accuracy on CIFAR-10 (11.11% improvement over original FF)

## Requirements

- PyTorch >= 1.7.0
- torchvision
- numpy
- matplotlib (for visualization)

## Usage

```python
# Create configuration
config = FF_DTG_Config()
config.input.dataset = "cifar10"  # Choose dataset
config.model.num_layers = 4       # Set architecture depth

# Create model
model = FF_DTG_Model(config)

# Training loop (example)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config.training.learning_rate,
    weight_decay=config.training.weight_decay
)

for epoch in range(config.training.epochs):
    # Your training loop implementation
    # ...
    outputs = model(batch_inputs, batch_labels)
    loss = outputs["Loss"]
    loss.backward()
    optimizer.step()
```

## Citation

If you use this code, please cite the original paper:

```
@article{DTG-FF-2025,
  title={Dynamic Temperature Goodness for Enhanced Forward-Forward Learning},
  author={Anonymous},
  booktitle={ICCV},
  year={2025}
}
```