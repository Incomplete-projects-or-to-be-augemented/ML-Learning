import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, all_hidden_states, current_query_source):
        # all_hidden_states: shape (batch, seq_length, hidden_size) -- every timestep's hidden state
        # current_query_source: shape (batch, hidden_size) -- typically the final hidden state

        query = self.query_layer(current_query_source)          # (batch, hidden_size)
        keys = self.key_layer(all_hidden_states)                # (batch, seq_length, hidden_size)
        values = self.value_layer(all_hidden_states)            # (batch, seq_length, hidden_size)

        query = query.unsqueeze(1)                               # (batch, 1, hidden_size)

        scores = torch.bmm(query, keys.transpose(1, 2))          # (batch, 1, seq_length)

        attention_weights = F.softmax(scores, dim=-1)             # (batch, 1, seq_length)

        context_vector = torch.bmm(attention_weights, values)      # (batch, 1, hidden_size)
        context_vector = context_vector.squeeze(1)                  # (batch, hidden_size)

        return context_vector, attention_weights.squeeze(1)
    
class TrendLSTMWithAttention(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        all_hidden_states, (final_hidden_state, final_cell_state) = self.lstm(x)
        # all_hidden_states: (batch, seq_length, hidden_size) -- every timestep, kept
        final_hidden = final_hidden_state.squeeze(0)             # (batch, hidden_size)

        context_vector, attention_weights = self.attention(all_hidden_states, final_hidden)

        prediction = self.fc(context_vector)                      # uses attention's output, not raw final_hidden
        return prediction, attention_weights
    
    
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
    
    
# (TrendDataset from before, unchanged)
dataset = TrendDataset(num_samples=500, seq_length=10)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---- Model ----
model = TrendLSTMWithAttention(input_size=1, hidden_size=16, num_classes=2)

# ---- Loss + optimizer ----
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---- Training loop ----
num_epochs = 30

for epoch in range(num_epochs):
    total_loss = 0
    for batch_sequences, batch_labels in data_loader:

        predictions, attention_weights = model(batch_sequences)   # note: two return values now

        loss = loss_function(predictions, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 5 == 0:
        average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch}, Average Loss: {average_loss:.4f}")