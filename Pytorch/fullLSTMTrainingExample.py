import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- Generate synthetic sequence data ----

class TrendDataset(Dataset):
    def __init__(self, num_samples, seq_length=10):
        self.sequences = []
        self.labels = []

        for _ in range(num_samples):
            if torch.rand(1).item() > 0.5:
                # upward trend + noise
                base = torch.linspace(0, 1, seq_length)
                label = 1
            else:
                # downward trend + noise
                base = torch.linspace(1, 0, seq_length)
                label = 0

            noisy_sequence = base + torch.randn(seq_length) * 0.1
            self.sequences.append(noisy_sequence.unsqueeze(1))  # shape (seq_length, 1)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]

dataset = TrendDataset(num_samples=500, seq_length=10)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---- Model ----

class TrendLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        last_hidden = final_hidden_state.squeeze(0)  # shape (batch, hidden_size)
        prediction = self.fc(last_hidden)
        return prediction

model = TrendLSTM()

# ---- Loss + optimizer ----

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---- Training loop ----

num_epochs = 30

for epoch in range(num_epochs):
    total_loss = 0
    for batch_sequences, batch_labels in data_loader:

        predictions = model(batch_sequences)

        loss = loss_function(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 5 == 0:
        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch}, Average Loss: {average_loss:.4f}")