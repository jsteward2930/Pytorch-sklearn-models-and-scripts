# Pytorch-SKlearn-models-and-scripts

## Overview
This repository contains implementations of various deep learning models and useful PyTorch scripts. It serves as a portfolio demonstrating proficiency in PyTorch and deep learning concepts. Each model is implemented with clean, well-documented code and includes training utilities, visualization tools, and example usage.

## Current Models
### Sparse Autoencoder
A complete implementation of a Sparse Autoencoder with the following features:
- Custom loss function combining reconstruction loss and KL divergence
- TensorBoard integration for training visualization
- Google Drive model persistence
- MNIST dataset training example
- Comprehensive training metrics tracking

#### Key Features:
- Sparsity constraint using KL divergence
- Customizable architecture (encoding size, sparsity parameter)
- Training progress visualization
- Model weights saving and loading utilities
- Performance metrics logging

#### Technical Details:
- Architecture: Fully connected layers with sigmoid activation
- Loss Function: MSE + KL divergence sparsity penalty
- Optimizer: Adam
- Training Metrics:
  - Reconstruction Loss
  - Sparsity Loss
  - Total Loss
  - Test Performance Metrics

## Repository Structure
```
├── models/
│   └── sparse_autoencoder/
│       ├── model.py
│       ├── train.py
│       └── utils.py
├── scripts/
│   ├── tensorboard_utils.py
│   └── drive_utils.py
├── notebooks/
│   └── sparse_autoencoder_demo.ipynb
└── README.md
```

## Dependencies
- PyTorch 2.0+
- TensorBoard
- NumPy
- tqdm
- Google Colab (for Google Drive integration)

## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/YourUsername/Pytorch-models-and-scripts.git
cd Pytorch-models-and-scripts
```

2. Install dependencies:
```bash
pip install torch torchvision tensorboard numpy tqdm
```

3. Run example notebook:
```bash
jupyter notebook notebooks/sparse_autoencoder_demo.ipynb
```

## Usage Examples
### Training a Sparse Autoencoder
```python
from models.sparse_autoencoder.model import SparseAutoencoder
from models.sparse_autoencoder.train import train_model

# Initialize model
model = SparseAutoencoder(
    input_size=784,
    encoding_size=128,
    sparsity_param=0.05,
    beta=3.0
)

# Train model
trained_model = train_model(
    model=model,
    train_loader=train_loader,
    num_epochs=50,
    learning_rate=0.001
)
```

## Model Performance
### Sparse Autoencoder Results
- Final Reconstruction Loss: 0.5422
- Final Sparsity Loss: 0.0937
- Total Loss: 0.6360
- Test Reconstruction Loss: 0.5486
- Test Sparsity Loss: 0.2713

## Future Additions
Planning to add implementations of:
- Variational Autoencoder (VAE)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transformer architectures
- More advanced training utilities

## Contributing
Feel free to open issues or submit pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
- GitHub: [Your GitHub Profile]
- LinkedIn: [Your LinkedIn Profile]
- Email: [Your Professional Email]

## Acknowledgments
- PyTorch Documentation and Tutorials
- Deep Learning Literature and Research Papers
- Open Source Community

---
**Note**: This repository is actively maintained and updated with new models and improvements.
