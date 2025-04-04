import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


# Patience = 150 triggered early stopping after 237 epochs
#   So model trained for 87 epochs
# LR = 0.001, no weight decay, Adam optimizer
# criterion = nn.BCEWithLogitsLoss()  

# My model
def init_weights(m): # # Xavier Initialization Function
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class TruthSeekerNet(nn.Module):
    def __init__(self):
        super(TruthSeekerNet, self).__init__()

        # CNN Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  

        # Compute CNN output size dynamically
        self._to_linear = None
        self._compute_linear_input_size()

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=self._to_linear, hidden_size=64, num_layers=1, batch_first=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64, 32)  # First hidden layer
        self.fc2 = nn.Linear(32, 1)  # Output layer

    def _compute_linear_input_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 5, 90)  # Dummy input
            x = self.pool(self.relu(self.conv1(x)))  
            x = self.dropout(x)
            x = self.pool(self.relu(self.conv2(x)))  
            x = self.dropout(x)
            x = x.permute(0, 2, 1, 3).contiguous()  # (batch, time_steps, channels, features)
            print(x.shape)
            x = x.view(x.size(0), x.size(1), -1)  # Flatten last two dimensions
            print(x.shape[2])
            self._to_linear = x.shape[2]  # Feature size per timestep for LSTM

    def forward(self, x):
        x = x.view(-1, 1, 5, 90)  # Ensure correct shape
        x = self.pool(self.relu(self.conv1(x)))  
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))  
        x = self.dropout(x)

        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, time_steps, channels, features)
        x = x.view(x.size(0), x.size(1), -1)  # Flatten (batch, time_steps, features)

        # LSTM processing
        x, _ = self.lstm(x)  # Output shape: (batch, time_steps, hidden_size)
        x = x[:, -1, :]  # Take last time step's output

        # Fully Connected Layers
        x = self.relu(self.fc1(x))  # First FC layer with activation
        x = self.dropout(x)
        x = self.fc2(x)  # Output layer
        return x  # No sigmoid (for regression tasks)


def train_and_validate(model, train_loader, val_loader, epochs=100, patience=20, print_every=5):
    torch.manual_seed(1000)
    model.train()
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")  # Initialize best validation loss
    best_val_acc = 0
    patience_counter = 0  # Counts epochs without improvement
    best_model_state = None  # Store best model weights

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.view(-1, 1).float()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))  
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # Validation loop (no gradient updates)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                labels = labels.view(-1, 1).float()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs))
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total

        # Print results
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%\n")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss  # Update best loss
            patience_counter = 0  # Reset counter
            best_model_state = model.state_dict()  # Save best model
        else:
            patience_counter += 1  # Increment counter

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break  # Stop training

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy

    # Load best model before exiting
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model with lowest validation loss.")

    print(f"Best val accuracy: {best_val_acc}")
    print(f"Best val loss: {best_val_loss}")

    # Plot the curves
    import matplotlib.pyplot as plt
    plt.title("Train vs Validation Accuracy")
    n = len(train_accuracies) # number of epochs
    plt.plot(range(1,n+1), train_accuracies, label="Train")
    plt.plot(range(1,n+1), val_accuracies, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_losses, label="Train")
    plt.plot(range(1,n+1), val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    return model  # Return best model


model1 = TruthSeekerNet()