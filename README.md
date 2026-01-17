# PyTorch_learningBook
Detailed deep learning using PyTorch

📘 PyTorch Basics – Clear & Descriptive Guide
Introduction

PyTorch is a widely used deep learning framework that allows developers to build and train neural networks easily using Python. This document explains the most important PyTorch concepts that are required to understand and write basic deep learning code. These concepts form the foundation for advanced topics such as CNNs, NLP, and Generative AI.

1. Tensor

A tensor is the core data structure in PyTorch. All data in PyTorch—such as numbers, images, text embeddings, and model weights—is represented as tensors. A tensor is similar to a NumPy array but has two major advantages: it can run on a GPU for faster computation, and it supports automatic gradient calculation.

In practice, tensors are used to store input data, output labels, and trainable parameters of a model.

x = torch.tensor([1.0, 2.0, 3.0])

2. requires_grad

The requires_grad property tells PyTorch whether it should track operations on a tensor. When this is set to True, PyTorch remembers all mathematical operations performed on that tensor so that gradients can be calculated later.

This is essential for training neural networks, because gradients are needed to update model weights during learning.

x = torch.tensor(2.0, requires_grad=True)

3. Autograd

Autograd is PyTorch’s automatic differentiation system. It automatically computes gradients for tensors that have requires_grad=True. Instead of manually calculating derivatives, PyTorch builds a computation graph during the forward pass and uses it to compute gradients during the backward pass.

Autograd makes deep learning practical and efficient.

y.backward()

4. Gradient

A gradient represents how much a value changes when a parameter changes slightly. In deep learning, gradients indicate how model weights should be adjusted to reduce prediction error.

Gradients are calculated during backpropagation and used by optimizers to improve the model.

x.grad

5. Neural Network

A neural network is a mathematical model made up of layers that learns patterns from data. Each layer applies transformations to the input, and the network gradually learns by adjusting its internal parameters using gradients.

Neural networks are used in image recognition, text processing, speech recognition, and many other AI applications.

6. nn.Module

nn.Module is the base class for all neural network models in PyTorch. Any model you create must inherit from this class. It provides important features such as parameter tracking, easy model saving/loading, and GPU support.

Without nn.Module, PyTorch cannot manage your model properly.

class SimpleNN(nn.Module):
    pass

7. Layer

A layer is a building block of a neural network. It takes input data, applies a mathematical operation, and produces output data. Different layers perform different tasks, such as learning linear relationships or extracting features.

Layers are stacked together to form a complete neural network.

8. nn.Linear

nn.Linear is a fully connected layer that performs a linear transformation on the input. It applies the equation:

y = weight × input + bias

This type of layer is commonly used in regression and classification tasks.

nn.Linear(1, 1)

9. forward() Method

The forward() method defines how input data flows through the neural network. It specifies the sequence of layers and operations applied to the input.

When you call the model like a function, PyTorch automatically executes the forward() method.

def forward(self, x):
    return self.linear(x)

10. Loss Function

A loss function measures how far the model’s predictions are from the actual target values. It provides a numerical value that represents model error.

The goal of training is to minimize this loss.

loss_fn = nn.MSELoss()

11. Optimizer

An optimizer is responsible for updating model weights based on gradients computed during backpropagation. It controls how learning happens and how fast the model improves.

Optimizers play a crucial role in training performance and stability.

optimizer = optim.SGD(model.parameters(), lr=0.01)

12. Learning Rate

The learning rate determines the step size used by the optimizer when updating weights. A learning rate that is too high can cause unstable training, while a very small learning rate can make training slow.

Choosing the right learning rate is essential for effective learning.

13. Training Loop

The training loop is the repeated process of teaching the model using data. In each iteration, the model makes predictions, calculates loss, computes gradients, and updates its weights.

This loop is the core of every deep learning program.

14. Epoch

An epoch refers to one complete pass of the entire training dataset through the model. Training usually requires multiple epochs for the model to learn meaningful patterns.

15. GPU (CUDA)

A GPU (Graphics Processing Unit) accelerates deep learning computations by performing many operations in parallel. PyTorch supports GPU acceleration through CUDA, which significantly reduces training time for large models.

16. Model Saving

Model saving allows you to store trained model parameters so they can be reused later without retraining. This is important for deployment and future experiments.

torch.save(model.state_dict(), "model.pth")

Final Concept Summary

Deep learning in PyTorch follows a simple and consistent workflow:

Forward Pass → Loss Calculation → Backward Pass → Weight Update

Understanding this flow means you understand the foundation of PyTorch.
