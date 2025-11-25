import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise

class TemporalJitterAugmentation(nn.Module):
    """
    Biologically-inspired temporal jitter augmentation for neural time-series.

    Biological Motivation:
    - Real neurons exhibit spike timing variability (±5-15ms jitter)
    - Networks trained with temporal perturbations are more robust
    - Inspired by Nature Communications 2024 research on SNNs

    Randomly shifts time-series features by ±jitter_range timesteps during training.
    Forces the network to learn time-invariant representations.

    Args:
        jitter_range (int): Maximum shift in timesteps (e.g., 2 means ±2 timesteps)
        p (float): Probability of applying jitter (default: 1.0, always apply)
    """
    def __init__(self, jitter_range=2, p=1.0):
        super().__init__()
        self.jitter_range = jitter_range
        self.p = p

    def forward(self, x):
        """
        Args:
            x: (batch, time, features) tensor
        Returns:
            Temporally jittered features
        """
        if not self.training or self.jitter_range == 0:
            return x

        # Apply jitter with probability p
        if torch.rand(1).item() > self.p:
            return x

        batch_size, seq_len, n_features = x.shape
        device = x.device

        # Random shift per sample in batch: uniform(-jitter_range, +jitter_range)
        shifts = torch.randint(
            -self.jitter_range,
            self.jitter_range + 1,
            (batch_size,),
            device=device
        )

        # Apply shifts using roll (circular shift to avoid boundary issues)
        jittered = torch.zeros_like(x)
        for i in range(batch_size):
            jittered[i] = torch.roll(x[i], shifts=shifts[i].item(), dims=0)

        return jittered


class NeuralGainModulation(nn.Module):
    """
    Biologically-inspired neural gain modulation for context-dependent processing.

    Biological Motivation:
    - Cortical neurons modulate gain based on attention/arousal states
    - Context-dependent gating enables multitask learning (2024 SNN research)
    - Inspired by prefrontal cortex context gating mechanisms

    Applies learnable day-specific (or context-specific) gain to scale features.
    Mathematically: output = gain[context] ⊙ input

    Args:
        hidden_dim (int): Feature dimension
        nContexts (int): Number of contexts (e.g., nDays=24)
        init_gain (float): Initial gain value (default: 1.0)
    """
    def __init__(self, hidden_dim, nContexts, init_gain=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nContexts = nContexts

        # Learnable per-context gain: (nContexts, hidden_dim)
        # Initialize near 1.0 for stability
        self.gain = nn.Parameter(torch.ones(nContexts, hidden_dim) * init_gain)

    def forward(self, x, context_idx):
        """
        Args:
            x: (batch, time, hidden_dim) features
            context_idx: (batch,) context indices (e.g., day indices)
        Returns:
            Gain-modulated features
        """
        # Select gain for each sample: (batch, hidden_dim)
        batch_gain = torch.index_select(self.gain, 0, context_idx)

        # Apply softplus to ensure positive gains (stable gradients)
        batch_gain = F.softplus(batch_gain)

        # Broadcast and multiply: (batch, time, hidden_dim)
        # batch_gain is (batch, 1, hidden_dim) after unsqueeze
        return x * batch_gain.unsqueeze(1)


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")
