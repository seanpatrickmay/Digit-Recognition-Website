import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
print(f"Using {device} as device")

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.convolutional_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_stack = nn.Sequential(
            nn.Linear(7 * 7 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.convolutional_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

convolutional_model = ConvolutionalNeuralNetwork().to(device)
print(convolutional_model)

loss_fn = nn.CrossEntropyLoss()
convolutional_optimizer = torch.optim.Adam(convolutional_model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch - 1) % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    average_loss = test_loss / num_batches
    accuracy = correct / size
    print(f"Accuracy: {accuracy}")
    print(f"Average Loss: {average_loss}")
    return accuracy, average_loss

MAX_EPOCHS = 20

def train_epochs(epochs, model, loss_fn, optimizer, forgiveness=1, minimum_epochs=3):
    losses = []
    accuracies = []
    average_loss_min = float('inf')
    misses = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n---------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy, average_loss = test(train_dataloader, model, loss_fn)
        losses.append(average_loss)
        accuracies.append(accuracy)
        print(f"Epoch Average Loss: {average_loss}, Min Average Loss: {average_loss_min}")
        print(f"Is Epoch Avg < Min Avg?: {average_loss < average_loss_min}")
        if average_loss < average_loss_min:
            average_loss_min = average_loss
            torch.save(convolutional_model.state_dict(), "mid_epoch_convolutional_model.pth")
            misses = 0
        elif epoch > minimum_epochs:
            print(f"Loss increased after {epoch+1} epochs")
            misses += 1
            if misses > forgiveness:
                print("miss!")
                # Reload the last state, before epoch that degraded model
                new_convolutional_model = ConvolutionalNeuralNetwork().to(device)
                new_convolutional_model.load_state_dict(torch.load("mid_epoch_convolutional_model.pth", weights_only = False))
                break
        # Save Model after each epoch, to revert to last increasing Model
    return losses, accuracies

RETRAIN_MODELS = True

if RETRAIN_MODELS:
    convolutional_losses, convolutional_accuracies = train_epochs(MAX_EPOCHS, convolutional_model, loss_fn, convolutional_optimizer)

    torch.save(convolutional_model.state_dict(), "convolutional_model.pth")
else:
    convolutional_model = ConvolutionalNeuralNetwork().to(device)
    convolutional_model.load_state_dict(torch.load("convolutional_model.pth", weights_only=True))

convolutional_accuracy, convolutional_loss = test(test_dataloader, convolutional_model, loss_fn)

print(f"Convolutional model accuracy: {convolutional_accuracy}")

def save_accuracy_loss_graph(name, accuracy, loss):
    accuracy_x = [num + 1 for num in range(len(accuracy))]
    loss_x = [num_loss + 1 for num_loss in range(len(loss))]
    plt.plot(accuracy_x, accuracy, label='Accuracy', color='blue')
    plt.plot(loss_x, loss, label='Loss', color='red')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(name + '_accuracy_loss_graph.png')
    plt.close()

print(convolutional_accuracies, convolutional_losses)

save_accuracy_loss_graph('convolutional', convolutional_accuracies, convolutional_losses)
