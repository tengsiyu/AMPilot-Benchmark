import torch
import torch.nn as nn


class BiIFNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, future_steps=6):
        super(BiIFNet, self).__init__()
        self.future_steps = future_steps

        # Bidirectional GRU encoder on inputs (B, T, C)
        self.bigru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Fuse the concatenated hidden features from two directions
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Decoder GRU that rolls out future steps from the fused context
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (B, 12, 2)
        bi_out, _ = self.bigru(x)  # (B, T, 2*hidden)
        context = bi_out[:, -1, :]  # (B, 2*hidden)
        # Fuse and repeat across future steps as decoder input
        fused = self.fusion(context).unsqueeze(1).repeat(1, self.future_steps, 1)  # (B, S, hidden)
        dec_out, _ = self.decoder(fused)  # (B, S, hidden)
        out = self.output(dec_out)  # (B, S, 2)
        return out
