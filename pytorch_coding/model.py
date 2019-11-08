import torch
import torch.nn as nn

import numpy as np


class EncoderRNN(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.gru = nn.GRU(inputs_size, hidden_size)

    def forward(self, inputs):
        # inputs: (sequence, batch, feature)
        # outputs: (sequence, batch, hidden)
        # hidden: (layer, batch, hidden)

        outputs, hidden = self.gru(inputs)
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRUCell(inputs_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs, hidden):
        # inputs: (batch, feature)
        # hidden: (batch, hidden)
        # output: (batch, 1)

        hidden = self.gru(inputs, hidden)
        output = self.linear(hidden)
        return hidden, output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, inputs, outputs, mean_, std_, teacher_forcing_ratio=0.5):
        # inputs: (input_sequence, batch, feature)
        # outputs: (output_sequence, batch, feature)
        # targets: (output_sequence, batch, 1)
        output_sequence_size, batch_size, feature_size = outputs.size()
        encoder_hidden = self.encoder(inputs)

        decoder_input = torch.cat([inputs[-1][:, :1], outputs[0][:, 1:]], dim=1)
        decoder_hidden = encoder_hidden[0]
        decoder_outputs = torch.zeros(output_sequence_size, batch_size, 1).to(self.device)
        for i in range(output_sequence_size):
            decoder_hidden, decoder_output = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs[i] = decoder_output

            if i == output_sequence_size - 1:
                break

            if np.random.random() < teacher_forcing_ratio:
                decoder_input = torch.cat([outputs[i][:, :1], outputs[i+1][:, 1:]], dim=1)
            else:
                # no teacher forcing
                decoder_output = decoder_output - mean_ / std_
                decoder_input = torch.cat([decoder_output, outputs[i+1][:, 1:]], dim=1)

        return decoder_outputs