Documentation for Pytorch, Sklearn Models and Scripts
Overview
## Overview

This repository contains implementations of various deep learning models and useful PyTorch scripts, serving as a portfolio that demonstrates proficiency in PyTorch and deep learning concepts. Each model is implemented with clean, well-documented code and includes training utilities, visualization tools, and example usage.

### Current Models

#### Sparse Autoencoder
A complete implementation of a Sparse Autoencoder featuring:
- A custom loss function that combines reconstruction loss and KL divergence.
- TensorBoard integration for training visualization.
- Google Drive model persistence.
- An example of training on the MNIST dataset.
- Comprehensive tracking of training metrics.

##### Key Features:
- Sparsity constraint using KL divergence.
- Customizable architecture (encoding size, sparsity parameter).
- Visualization of training progress.
- Utilities for saving and loading model weights.
- Logging of performance metrics.

##### Technical Details:
- **Architecture**: Fully connected layers with sigmoid activation.
- **Loss Function**: Mean Squared Error (MSE) combined with KL divergence sparsity penalty.
- **Optimizer**: Adam.
- **Training Metrics**:
  - Reconstruction Loss
  - Sparsity Loss
  - Total Loss
  - Test Performance Metrics

This repository is actively maintained and updated with new models and improvements, with plans to add implementations of Variational Autoencoders (VAEs), Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformer architectures, and more advanced training utilities in the future.
Installation
Installation
To set up the repository and run the models, follow the steps outlined below:
Prerequisites
Ensure you have the following installed on your system:
Python (version 3.6 or higher recommended)
Git (to clone the repository)
Steps to Install
Clone the Repository
Open your terminal and run the following command to clone the repository:
git clone https://github.com/YourUsername/Pytorch-models-and-scripts.git
Navigate into the cloned directory:
cd Pytorch-models-and-scripts
Install Dependencies
Use pip to install the required packages. Run the following command:
pip install torch torchvision tensorboard numpy tqdm
Ensure you have PyTorch 2.0+ installed. You can verify the installation by running:
python -c "import torch; print(torch.__version__)"
Run Example Notebook
To start using the models, launch Jupyter Notebook and open the example notebook:
jupyter notebook notebooks/sparse_autoencoder_demo.ipynb
This notebook provides a demonstration of how to train the Sparse Autoencoder model.
Additional Notes
For Google Drive integration, it is recommended to use Google Colab.
Make sure to check the README.md file for further instructions and usage examples.
By following these steps, you will have the repository set up and ready for experimentation with the models provided.
Usage
## Usage

This section provides guidance on how to utilize the models and scripts available in this repository. Below are examples of how to train a Sparse Autoencoder using the provided code.

### Training a Sparse Autoencoder

To train a Sparse Autoencoder, you can follow the example code below. This code initializes the model and trains it using a specified training dataset.

``python
from models.sparse_autoencoder.model import SparseAutoencoder
from models.sparse_autoencoder.train import train_model

# Initialize model
model = SparseAutoencoder(
    input_size=784,        # Input size for MNIST images (28x28)
    encoding_size=128,     # Size of the encoding layer
    sparsity_param=0.05,   # Sparsity parameter for KL divergence
    beta=3.0               # Weight for the sparsity loss
)

# Train model
trained_model = train_model(
    model=model,
    train_loader=train_loader,  # DataLoader for training data
    num_epochs=50,              # Number of training epochs
    learning_rate=0.001         # Learning rate for the optimizer
)
Example Notebook
For a more detailed walkthrough, you can run the example notebook provided in the notebooks directory. This notebook demonstrates the complete workflow of training the Sparse Autoencoder, including data loading, model training, and performance evaluation.
To run the example notebook, execute the following command in your terminal:
jupyter notebook notebooks/sparse_autoencoder_demo.ipynb
Additional Notes
Ensure that you have the required dependencies installed as mentioned in the Getting Started section.
Modify the parameters in the model initialization and training function as needed to suit your specific dataset or requirements.
Monitor the training process using TensorBoard for visual insights into the model's performance.
By following these instructions, you can effectively utilize the Sparse Autoencoder model for your own projects or experiments.

## Current Models

``markdown
## Current Models

### Sparse Autoencoder
A complete implementation of a Sparse Autoencoder with the following features:
- Custom loss function combining reconstruction loss and KL divergence
- TensorBoard integration for training visualization
- Google Drive model persistence
- MNIST dataset training example
- Comprehensive training metrics tracking

#### Key Features:
- **Sparsity Constraint**: Utilizes KL divergence to enforce sparsity in the learned representations.
- **Customizable Architecture**: Allows adjustments to encoding size and sparsity parameters for tailored model performance.
- **Training Progress Visualization**: Integrated TensorBoard support for real-time monitoring of training metrics.
- **Model Weights Management**: Utilities for saving and loading model weights to and from Google Drive.
- **Performance Metrics Logging**: Detailed tracking of various training metrics for analysis.

#### Technical Details:
- **Architecture**: Composed of fully connected layers with sigmoid activation functions.
- **Loss Function**: Combines Mean Squared Error (MSE) with a KL divergence sparsity penalty.
- **Optimizer**: Adam optimizer is employed for efficient training.
- **Training Metrics**:
  - Reconstruction Loss
  - Sparsity Loss
  - Total Loss
  - Test Performance Metrics

### Additional Models
In addition to the Sparse Autoencoder, the repository includes implementations of various other models, such as:
- **K-means Clustering for Cancer Text Classification**
- **Medical Text Classification using CNN (PyTorch)**
- **Medical Text Classification using SVM**
- **Perceptron (Scikit-learn)**
- **Logistic Regression for Breast Cancer Classification**
- **Gradient Descent Logistic Regression**

Each model is designed to demonstrate different aspects of machine learning and deep learning techniques, providing a comprehensive portfolio of implementations.
Key Features
## Key Features

This repository showcases a variety of deep learning models and scripts, each with unique functionalities. Below are the key features of the primary models included:

### Sparse Autoencoder
- **Custom Loss Function**: Combines reconstruction loss with KL divergence to enforce sparsity in the learned representations.
- **TensorBoard Integration**: Provides real-time training visualization, allowing users to monitor training progress and performance metrics.
- **Model Persistence**: Supports saving and loading of model weights to and from Google Drive, facilitating easy access and sharing.
- **Comprehensive Training Metrics**: Tracks and logs essential metrics during training, including reconstruction loss, sparsity loss, and total loss.

#### Additional Features:
- **Customizable Architecture**: Users can modify the encoding size and sparsity parameter to tailor the model to specific needs.
- **Training Progress Visualization**: Visual feedback on training dynamics through TensorBoard.
- **Performance Metrics Logging**: Detailed logging of training and test performance metrics for thorough evaluation.

### Other Models
- **Medical Text Classification**: Implementations using CNN and SVM for classifying medical texts, showcasing versatility in model architecture and application.
- **Logistic Regression**: Includes implementations for breast cancer classification using logistic regression techniques.
- **Perceptron Model**: A basic implementation of the perceptron algorithm using scikit-learn, demonstrating foundational machine learning concepts.

This repository serves as a comprehensive portfolio for exploring various deep learning methodologies and their applications in medical text classification and other domains.
Technical Details
## Technical Details

### Sparse Autoencoder
The Sparse Autoencoder implementation in this repository is designed to effectively learn representations of input data while enforcing sparsity in the hidden layer activations. Below are the key technical specifications of the model:

#### Architecture
- **Type**: Fully connected neural network
- **Activation Function**: Sigmoid
- **Layers**: 
  - Input layer with 784 neurons (for MNIST images)
  - Hidden layer with customizable encoding size (default: 128)
  - Output layer matching the input size

#### Loss Function
- **Components**:
  - Mean Squared Error (MSE) for reconstruction loss
  - Kullback-Leibler (KL) divergence for sparsity penalty
- **Total Loss**: 
  - Combined loss = MSE + KL divergence

#### Optimizer
- **Type**: Adam optimizer
- **Learning Rate**: Customizable (default: 0.001)

#### Training Metrics
During training, the following metrics are tracked:
- **Reconstruction Loss**: Measures how well the model reconstructs the input data.
- **Sparsity Loss**: Quantifies the degree of sparsity enforced on the hidden layer activations.
- **Total Loss**: The overall loss combining reconstruction and sparsity losses.
- **Test Performance Metrics**: Evaluates the model's performance on a separate test dataset.

### Dependencies
To ensure proper functionality, the following libraries are required:
- **PyTorch**: Version 2.0 or higher
- **TensorBoard**: For visualization of training progress
- **NumPy**: For numerical operations
- **tqdm**: For progress bar functionality
- **Google Colab**: Recommended for Google Drive integration and ease of use

### Model Persistence
- **Google Drive Integration**: The model supports saving and loading weights directly to and from Google Drive, facilitating easy access and persistence across sessions.

### Visualization
- **TensorBoard Integration**: Training progress, including loss metrics, is visualized using TensorBoard, allowing for real-time monitoring of the training process.

### Example Usage
A sample code snippet for training the Sparse Autoencoder is provided in the repository, demonstrating how to initialize the model and execute the training process.

This implementation serves as a robust foundation for further experimentation and development in the field of deep learning, particularly in unsupervised learning tasks.
Repository Structure
## Repository Structure

This repository is organized into several key directories and files, each serving a specific purpose. Below is an overview of the structure:

├── notebooks/ │ ├── K_means_Clustering_for_Cancer_Text_Classification_1_15_25.ipynb │ ├── Medical_Text_Classification_CNN_using_PyTorch_1_14_25.ipynb │ ├── Medical_Text_Classification_using_SVM_1_13_24.ipynb │ ├── Perceptron_scikit_learn_1_11_25.ipynb │ ├── gradient_descent_logistic_regression_1_11_24.ipynb │ ├── logistic_regression_Breast_Cancer_Wisconsin_dataset_1_11_24.ipynb │ └── sparse_autoencoder_MNIST_tensorboard_11_22_24.ipynb └── README.md

### Directory and File Descriptions

- **notebooks/**: This directory contains Jupyter notebooks that demonstrate various machine learning models and techniques. Each notebook is named to reflect its content and includes implementations for different classification tasks and algorithms.

  - `K_means_Clustering_for_Cancer_Text_Classification_1_15_25.ipynb`: Implementation of K-means clustering for cancer text classification.
  - `Medical_Text_Classification_CNN_using_PyTorch_1_14_25.ipynb`: A convolutional neural network model for medical text classification using PyTorch.
  - `Medical_Text_Classification_using_SVM_1_13_24.ipynb`: Support Vector Machine approach for medical text classification.
  - `Perceptron_scikit_learn_1_11_25.ipynb`: Implementation of a perceptron model using scikit-learn.
  - `gradient_descent_logistic_regression_1_11_24.ipynb`: Logistic regression model utilizing gradient descent.
  - `logistic_regression_Breast_Cancer_Wisconsin_dataset_1_11_24.ipynb`: Logistic regression applied to the Breast Cancer Wisconsin dataset.
  - `sparse_autoencoder_MNIST_tensorboard_11_22_24.ipynb`: Demonstration of a sparse autoencoder with TensorBoard integration for the MNIST dataset.

- **README.md**: The main documentation file for the repository, providing an overview, usage instructions, model descriptions, and other relevant information.

This structure is designed to facilitate easy navigation and understanding of the various models and scripts included in the repository.
Contributing
## Contributing

We welcome contributions to enhance the functionality and performance of this repository. Whether you have suggestions for improvements, bug fixes, or new features, your input is valuable. Please follow the guidelines below to contribute effectively:

### How to Contribute

1. **Fork the Repository**
   - Click the "Fork" button at the top right of the repository page to create your own copy of the repository.

2. **Clone Your Fork**
   - Clone your forked repository to your local machine:
     ```bash
     git clone https://github.com/YourUsername/Pytorch-models-and-scripts.git
     cd Pytorch-models-and-scripts
     ```

3. **Create a New Branch**
   - Create a new branch for your feature or bug fix:
     ```bash
     git checkout -b feature/your-feature-name
     ```

4. **Make Your Changes**
   - Implement your changes, ensuring that your code is clean, well-documented, and follows the existing coding style.

5. **Run Tests**
   - If applicable, run any existing tests to ensure your changes do not break functionality.

6. **Commit Your Changes**
   - Commit your changes with a descriptive message:
     ```bash
     git commit -m "Add feature: your feature description"
     ```

7. **Push to Your Fork**
   - Push your changes to your forked repository:
     ```bash
     git push origin feature/your-feature-name
     ```

8. **Submit a Pull Request**
   - Go to the original repository and click on "Pull Requests." Then click on "New Pull Request" and select your branch to submit your changes for review.

### Reporting Issues

If you encounter any bugs or have suggestions for improvements, please open an issue in the repository. Provide as much detail as possible, including steps to reproduce the issue, expected behavior, and any relevant screenshots or logs.

### Code of Conduct

Please adhere to our [Code of Conduct](#) while contributing to this project. We aim to create a welcoming and inclusive environment for all contributors.

### Acknowledgments

Thank you for considering contributing to this project! Your efforts help improve the repository and benefit the community.
License
## License

This project is licensed under the MIT License. The MIT License is a permissive free software license that allows for reuse within proprietary software, as long as the license is included with that software. 

### Summary of the MIT License:
- **Permission** is granted to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.
- **Attribution** must be given to the original authors of the software.
- **Warranty Disclaimer**: The software is provided "as is", without warranty of any kind, express or implied.
- **Liability Disclaimer**: The authors are not liable for any damages arising from the use of the software.

For more detailed information, please refer to the [LICENSE file](LICENSE) in this repository.

