# src/capsnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def squash(tensor, dim=-1, eps=1e-9):
    sq_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    scale = sq_norm / (1.0 + sq_norm)
    return scale * tensor / (torch.sqrt(sq_norm) + eps)

class PrimaryCaps1D(nn.Module):
    def __init__(self, in_dim, num_capsules=32, dim_caps=8):
        super().__init__()
        # a small linear -> reshape into capsules
        self.fc = nn.Linear(in_dim, num_capsules * dim_caps)
        self.num_capsules = num_capsules
        self.dim_caps = dim_caps

    def forward(self, x):
        # x: [B, F] feature vector
        out = self.fc(x)  # [B, num_capsules*dim]
        B = out.size(0)
        out = out.view(B, self.num_capsules, self.dim_caps)  # [B, num_capsules, dim]
        return squash(out)

class CapsBiLSTM(nn.Module):
    def __init__(self, feature_dim, n_classes, primary_caps=32, primary_dim=8, lstm_hidden=128, lstm_layers=1, dropout=0.3):
        super().__init__()
        self.primary = PrimaryCaps1D(in_dim=feature_dim, num_capsules=primary_caps, dim_caps=primary_dim)
        self.bilstm = nn.LSTM(input_size=primary_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
                              batch_first=True, bidirectional=True, dropout=dropout if lstm_layers>1 else 0.0)
        self.fc = nn.Linear(2*lstm_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, feature_dim]
        caps = self.primary(x)                # [B, seq_len=num_capsules, dim]
        out, (hn, cn) = self.bilstm(caps)     # out: [B, seq_len, 2*hidden]
        # use attention or last timestep; we'll use mean pooling across timesteps
        pooled = out.mean(dim=1)              # [B, 2*hidden]
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits
