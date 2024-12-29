import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score)
import time
import numpy as np
import pandas as pd
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import tensorflow as tf
import torch.nn as nn
from tensorflow import keras
from keras import layers, models
from tensorflow.keras.datasets import cifar10

# Set random seed for reproducibility, ability to achieve consistent results across multiple runs
torch.manual_seed(42)
np.random.seed(42) # If not use, Different runs may yield varying results, making it harder to compare models or debug issues


#######################
# Regression Task (PyTorch ANN)
#######################
# Load and preprocess the California Housing dataset
data = fetch_california_housing()
X = data.data #A matrix of size (n_samples, n_features)
y = data.target #The median house value for each sample

scaler = StandardScaler() #  mean =0, and SD=1
X_scaled = scaler.fit_transform(X)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42 #validation data tunes hyperparameters
)
#A tensor is a multi-dimensional array used in PyTorch for numerical computations
# Convert numpy array to PyTorch tensors,bcz PyTorch only works with tensors, if we not provide tensor, training process will braek
X_train_tensor = torch.tensor(X_train, dtype=torch.float32) #specify this to float to maintain consistency, although it default
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # 2D column vector
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1) #view is a PyTorch method to reshape a tensor without changing its data.
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the ANN model for regression
class RegressionANN(nn.Module):
    def __init__(self):
        super(RegressionANN, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) #activation func use to introduce non-linearity so that network learn complex patterns, 
        x = torch.relu(self.fc2(x))  #otherwise it will work as linear regression
        x = self.fc3(x)
        return x

model = RegressionANN()
criterion = nn.MSELoss() # Loss Func
optimizer = optim.Adam(model.parameters(), lr=0.001) 
#optimizer is used to update the weights of the network to minimize the loss function

num_epochs = 100
train_losses = []
val_losses = []

# Training loop for regression
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad() #resets gradients to zero before backpropagation
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor) #Refers to the loss function used to measure the difference between predicted and actual values
    train_losses.append(loss.item())
    loss.backward() #Computes the gradients using backpropagation
    optimizer.step() #Updates the weights of the network

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())

# Evaluate model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)

mse = mean_squared_error(y_test_tensor.numpy(), y_pred.numpy())
mae = mean_absolute_error(y_test_tensor.numpy(), y_pred.numpy())
r2 = r2_score(y_test_tensor.numpy(), y_pred.numpy())
print()
print("Regression Task Metrics (PyTorch ANN):")
print(f"MSE: {mse:.4f}, \nMAE: {mae:.4f}, \nR2: {r2:.4f}")


end_time = time.time()
pyTorch_regression_training_time = end_time - start_time
print(f"Total Training Time for PyTorch Regression: {pyTorch_regression_training_time:.2f} seconds")

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Learning Curves (Regression)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
print()
print()
print()
print()



#######################
# Classification Task (PyTorch ANN)
#######################
# Dataset and model setup for classification (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) #normalize data to have mean 0.5 and std 0.5)
    ])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

val_size = int(0.1 * len(train_dataset)) #validation data tunes hyperparameters,   10% of training data
train_size = len(train_dataset) - val_size #remaining 90% of training data
train_subset, val_subset = random_split(train_dataset, [train_size, val_size]) #split the training data into training and validation sets

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True) #shuffle=True to shuffle the data before each epoch
val_loader = DataLoader(val_subset, batch_size=64, shuffle=False) #no need to shuffle the validation data
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) #no need to shuffle the test data

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) #28*28 is the input size, 128 is the output size
        self.fc2 = nn.Linear(128, 64) 
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleANN()
criterion = nn.CrossEntropyLoss() #CrossEntropyLoss is used for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001) 

num_epochs = 10
train_accuracies = []
val_accuracies = []

# Training loop for classification
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader: #inputs are the images, labels are the corresponding labels
        optimizer.zero_grad() 
        outputs = model(inputs)
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 

        _, predicted = torch.max(outputs.data, 1) #torch.max returns the maximum value and its index,
         #outputs.data give predicted values, _ is used to ignore actual values
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracies.append(correct_train / total_train)

    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracies.append(correct_val / total_val)

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Learning Curves (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion Matrix and Metrics
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')# average='macro' computed for each class and then average
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Classification Task Metrics (PyTorch ANN):")
print(f"Accuracy: {accuracy:.4f}, \nPrecision: {precision:.4f}, \nRecall: {recall:.4f}, \nF1 Score: {f1:.4f}")

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix (MNIST)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Track time for PyTorch Classification (ANN)
end_time = time.time()
pyTorch_classification_training_time = end_time - start_time
print(f"Total Training Time for PyTorch Classification (ANN): {pyTorch_classification_training_time:.2f} seconds")
print()
print()
print()
print()


#######################
# Classification Task (Keras CNN)
#######################
start_time = time.time()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

cnn_model = models.Sequential([ #simplest way to build a CNN model in Keras
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)), #32 filters 3x3 kernel, 32x32 image size 3 channels
    layers.MaxPooling2D((2, 2)), #reduce complexity
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(), # flatten the output of the previous layer to a 1D array
    layers.Dense(64, activation='relu'), # adds fc layer with 64 neurons
    layers.Dense(10, activation='softmax') #converts raw prediction scores (logits) into probabilities
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#The loss function computes cross-entropy loss between predicted probabilities and integer class labels in multi-class classification.
history = cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy') #history.history is a dictionary containing metrics
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Learning Curves (CIFAR-10)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion Matrix and Metrics
cnn_y_pred = cnn_model.predict(x_test) #predicts the probabilities of each class
cnn_y_pred_classes = np.argmax(cnn_y_pred, axis=1) #argmax returns the index of the maximum value along an axis

accuracy = accuracy_score(y_test, cnn_y_pred_classes)
precision = precision_score(y_test, cnn_y_pred_classes, average='macro')
recall = recall_score(y_test, cnn_y_pred_classes, average='macro')
f1 = f1_score(y_test, cnn_y_pred_classes, average='macro')

print("Classification Task Metrics (Keras CNN):")
print(f"Accuracy: {accuracy:.4f}, \nPrecision: {precision:.4f}, \nRecall: {recall:.4f}, \nF1 Score: {f1:.4f}")

cm = confusion_matrix(y_test, cnn_y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix (CIFAR-10)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

end_time = time.time()
keras_cnn_training_time = end_time - start_time
print(f"Total Training Time for Keras CNN: {keras_cnn_training_time:.2f} seconds")
print()
print()
print()
print()

#######################
# Comparative Table
#######################

comparison_data = {
    "Model": ["PyTorch ANN (Regression)", "PyTorch ANN (Classification)", "Keras CNN (Classification)"],
    "Dataset/Task": ["California Housing", "MNIST", "CIFAR-10"],
    "Key Hyperparams": ["lr=0.001, batch_size=64, epochs=100", "lr=0.001, batch_size=64, epochs=10", "lr=0.001, batch_size=64, epochs=10"],
    "Final Metric": [f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}", f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}", f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}"],
    "Training Time": [f"{pyTorch_regression_training_time:.2f} seconds", f"{pyTorch_classification_training_time:.2f} seconds", f"{keras_cnn_training_time:.2f} seconds"]
}

df_comparison = pd.DataFrame(comparison_data)
print("\nComparative Table:")
print(df_comparison)

print()
print()
