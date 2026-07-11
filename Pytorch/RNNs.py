import torch
import torch.nn as nn

rnn_layer = nn.RNN(input_size=10, hidden_size=20, batch_first=True)

sequence = torch.rand(1, 5, 10)  # (batch, sequence_length, input_size)

output, final_hidden_state = rnn_layer(sequence)

print("output shape:", output.shape)
print("final hidden state shape:", final_hidden_state.shape)