import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

# ── 1. SETTINGS ───────────────────────────────────────────
TRAIN_DIR   = 'Emotions Pics/train'
TEST_DIR    = 'Emotions Pics/test'
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 0.001
NUM_CLASSES = 7

# ── 2. TRANSFORMS ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 3. LOAD DATASET ───────────────────────────────────────
print("Loading dataset...")
train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_data  = datasets.ImageFolder(TEST_DIR,  transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples : {len(train_data)}")
print(f"Testing samples  : {len(test_data)}")
print(f"Classes          : {train_data.classes}")

# ── 4. LOAD RESNET18 ──────────────────────────────────────
print("\nLoading ResNet18 model...")
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)

# ── 5. LOSS & OPTIMIZER ───────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# ── 6. TRAINING LOOP ──────────────────────────────────────
train_losses     = []
train_accuracies = []

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted   = outputs.argmax(1)
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)
        loop.set_postfix(loss=loss.item())

    epoch_loss = total_loss / len(train_loader)
    epoch_acc  = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# ── 7. TESTING ────────────────────────────────────────────
print("\nEvaluating on test set...")
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for images, labels in test_loader:
        images    = images.to(device)
        labels    = labels.to(device)
        outputs   = model(images)
        predicted = outputs.argmax(1)
        correct  += (predicted == labels).sum().item()
        total    += labels.size(0)

print(f"Test Accuracy: {100*correct/total:.2f}%")

# ── 8. SAVE MODEL ─────────────────────────────────────────
torch.save(model.state_dict(), 'face_model.pth')
print("Model saved as face_model.pth")