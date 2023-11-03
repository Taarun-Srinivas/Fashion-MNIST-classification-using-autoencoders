import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


torch.manual_seed(42)
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.FashionMNIST(root='./data',train=True,transform=transform,download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), #128
            nn.ReLU(),
            nn.Linear(128,64), #128, 64
            nn.ReLU(),
            nn.Linear(64,32), #64, 32
            nn.ReLU(),
            # nn.Linear(64,32),
            # nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch_data, _ in train_loader:
        optimizer.zero_grad()
        reconstructed_data = autoencoder(batch_data)
        loss = criterion(reconstructed_data, batch_data.view(batch_data.size(0), -1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}')


with torch.no_grad():
    example_batch, _ = next(iter(train_loader))
    reconstructed_batch = autoencoder(example_batch)

plt.figure(figsize=(8, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(example_batch[i].view(28, 28).numpy(), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed_batch[i].view(28, 28).numpy(), cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

plt.show()

test_dataset = datasets.FashionMNIST(root='./data', train = False, transform=transform)
batch_size = 64
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

encoded_representations = []
labels = []

autoencoder.eval()

with torch.no_grad():
    for data in test_loader:
        inputs, labels_batch = data
        encoded_batch = autoencoder.encoder(inputs.view(-1, 28 * 28))
        encoded_representations.extend(encoded_batch.numpy())
        labels.extend(labels_batch.numpy())

encoded_representations = np.array(encoded_representations)
tsne = TSNE(n_components=2, random_state=42)
encoded_2d = tsne.fit_transform(encoded_representations)

# Visualize the 2D representation
plt.scatter(encoded_2d[:, 0], encoded_2d[:, 1], c=labels, cmap='jet')
plt.title("2D Visualization of 10000 data samples")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar()
plt.show()
