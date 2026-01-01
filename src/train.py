from torchvision import transforms
from torch.utils.data import DataLoader

from mvtec_dataset import MVtecDataset


# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
   transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# Datasets
train_dataset = MVtecDataset(
    root_dir="src/../archive/capsule",
    split="train",
    transform=train_transforms
)

test_dataset = MVtecDataset(
    root_dir="src/../archive/capsule",
    split="test",
    transform=test_transforms
)


# Dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# test
print(f"Train samples: {len(train_dataset)}")
print(train_loader)
print(f"Test samples: {len(test_dataset)}")

# CNN Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()

    self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #64

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) #16
            )
    self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
            )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Loss and optimizer

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = opti.Adam(model.parameters(),lr=1e-3)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total

num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} "
        f"Test Acc: {test_acc:.3f}"
    )


images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

