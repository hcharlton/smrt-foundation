import torch
import torch.nn as nn

class SingleIdxProbe(nn.Module):
    def __init__(self, encoder, n_classes=1, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
          self.encoder.requires_grad_(False)
        else:
          self.encoder.requires_grad_(True)

        self.head = nn.Sequential(
            nn.Linear(encoder.d_model, encoder.d_model // 2),
            nn.ReLU(),
            nn.Linear(encoder.d_model // 2, n_classes)
        )

    def forward(self, x):
        c = self.encoder(x)
        logit = self.head(c[:, -1, :])
        return logit