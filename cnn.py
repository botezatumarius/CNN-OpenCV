import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the CNN model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # First convolutional layer (input: 3 channels, output: 16 filters, kernel size: 3x3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with 2x2 window
        
        # Second convolutional layer (input: 16 filters, output: 32 filters, kernel size: 3x3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Third convolutional layer (input: 32 filters, output: 64 filters, kernel size: 3x3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 31 * 31, 128)  # 64 channels, 31x31 image size after pooling
        self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification
    
    def forward(self, x):
        # Pass through convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output from convolutional layers to feed into fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        
        # Output layer with sigmoid activation for binary classification
        x = torch.sigmoid(self.fc2(x))
        
        return x

# Function to train the model
def train_model(train_loader, val_loader, epochs=5, lr=0.001):
    model = ConvNet()
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)
            
            # Calculate loss (for binary classification, outputs are logits)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimize the weights
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        # Validation step (optional)
        if val_loader:
            model.eval()  # Set model to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():  # Disable gradient calculation
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    predicted = (outputs > 0.5).float()  # Apply threshold (sigmoid > 0.5 -> True/False)
                    total += labels.size(0)
                    correct += (predicted.view(-1) == labels).sum().item()

            print(f"Validation Accuracy: {correct / total * 100}%")

    # Save the trained model
    torch.save(model.state_dict(), 'passport_model.pth')

# Data Preprocessing (Resizing images and normalization)
transform = transforms.Compose([
    transforms.Resize((250, 250)),  # Resize to 250x250
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
])

# Load dataset (assuming the images are in the 'images/' folder and labels are in 'labels.csv')
train_dataset = datasets.ImageFolder(root='train/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation set (if available)
val_dataset = datasets.ImageFolder(root='val/', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Train the model
train_model(train_loader, val_loader, epochs=5, lr=0.001)
