import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

training_data = datasets.FashionMNIST(root='fashion', train=True, transform=transform, download=True)
testing_data = datasets.FashionMNIST(root='fashion', train=False, transform=transform, download=True)

train_loader = DataLoader(training_data, 64, True)
test_loader = DataLoader(testing_data, 64, True)

fashion_mnist_classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

device = torch.device('mps')

class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),     
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)                    
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),              
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)                     
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 96, 3),               
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2)                 
        )

        self.fc = nn.Sequential(
            nn.Linear(96 * 2 * 2, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = FashionMNIST_CNN()
model.to(device)
print()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, 50, 0.1)
"""
model.train()
for epoch in range(30):
    running_loss = 0.0
    print(f'Training Epoch: {epoch}')
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        loss = loss_fn(output, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f'Loss: {running_loss / len(train_loader):.4f}')
    print()
    scheduler.step()

torch.save(model.state_dict(), 'trained_net.pth')
"""
model.load_state_dict(torch.load('trained_net.pth'))

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        pred = torch.argmax(output, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        for pred, actual in zip(pred, labels):
            pred_name = fashion_mnist_classes[pred.item()]
            actual_name = fashion_mnist_classes[actual.item()]
            print(f"Prediction: {pred_name}, Actual: {actual_name}")

print()
accuracy = 100 * correct / total
print(f'Accuracy: {accuracy}%')
