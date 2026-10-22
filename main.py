import torch
from torch import nn

torch.manual_seed(42)
num_samples = 1000
x = torch.randn(num_samples, 2)
distance = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
y = (distance >= 1.0).long()
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1:3d}, loss = {loss.item():.4f}")
with torch.no_grad():
    logits = model(x)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == y).float().mean()

print(f"训练集准确率：{accuracy.item() * 100:.2f}%")
