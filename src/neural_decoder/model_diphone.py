"""
Model 3: Baseline GRU + Artificial Drift (Channel Scaling and Shifting)

Implements channel-level augmentations to simulate recording drift:
- Channel Scaling: Multiply each channel by random scalar (0.9-1.1)
- Channel Shifting: Add random offset to each channel
"""

import torch
from torch import nn

from .augmentations import GaussianSmoothing


class GRUDiphoneDecoder(nn.Module):
    """
    Model 3: Baseline GRU + Artificial Drift (Channel Scaling and Shifting)

    Architecture: Same as baseline (Model 1)
    Augmentation: Channel-level drift simulation
    - Channel Scaling: Multiply each channel by 0.9-1.1
    - Channel Shifting: Add small random offset per channel

    This simulates recording drift and forces robustness to channel variations.
    """
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim=1024,
        layer_dim=5,
        nDays=24,
        dropout=0.4,
        device="cuda",
        strideLen=4,
        kernelLen=32,
        gaussianSmoothWidth=2.0,
        bidirectional=False,
        # Channel drift parameters
        channel_scale_min=0.9,
        channel_scale_max=1.1,
        channel_shift_std=0.1,
        drift_augment_prob=0.8,
        # Unused params (kept for compatibility)
        use_layer_norm=None,
        coordinated_dropout_p=None,
        return_diphones=None,
    ):
        super(GRUDiphoneDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        # Channel drift parameters
        self.channel_scale_min = channel_scale_min
        self.channel_scale_max = channel_scale_max
        self.channel_shift_std = channel_shift_std
        self.drift_augment_prob = drift_augment_prob

        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

    def _apply_channel_drift(self, neuralInput):
        """
        Apply channel scaling and shifting to simulate recording drift.

        Args:
            neuralInput: [batch, time, channels]

        Returns:
            Augmented input with same shape
        """
        if not self.training:
            return neuralInput

        # Only apply with probability drift_augment_prob
        if torch.rand(1).item() > self.drift_augment_prob:
            return neuralInput

        batch_size, time_steps, n_channels = neuralInput.shape
        device = neuralInput.device

        # Channel Scaling: Multiply each channel by random scalar (0.9-1.1)
        scale_factors = torch.FloatTensor(batch_size, 1, n_channels).uniform_(
            self.channel_scale_min, self.channel_scale_max
        ).to(device)

        # Channel Shifting: Add small random constant offset to each channel
        shift_offsets = torch.randn(batch_size, 1, n_channels, device=device) * self.channel_shift_std

        # Apply both augmentations
        augmented = neuralInput * scale_factors + shift_offsets

        return augmented

    def forward(self, neuralInput, dayIdx, return_diphones=False):
        """
        Forward pass with Artificial Drift (Channel Scaling and Shifting).
        """
        # Apply channel drift BEFORE any other processing
        neuralInput = self._apply_channel_drift(neuralInput)

        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
