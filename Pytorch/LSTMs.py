import torch
import torch.nn as nn

lstm_layer = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)

sequence = torch.rand(1, 5, 10)  # (batch, sequence_length, input_size)

output, (final_hidden_state, final_cell_state) = lstm_layer(sequence)

print("output shape:", output.shape)
print("final hidden state shape:", final_hidden_state.shape)
print("final cell state shape:", final_cell_state.shape)

print(lstm_layer.weight_ih_l0.shape)   # input-to-hidden weights, for all 4 gates stacked together
print(lstm_layer.weight_hh_l0.shape)   # hidden-to-hidden weights, for all 4 gates stacked together