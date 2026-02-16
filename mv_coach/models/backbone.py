"""Base backbone architectures for HAR."""

import torch
import torch.nn as nn
from typing import List


class TemporalBlock(nn.Module):
    """Temporal block for TCN with residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float,
    ) -> None:
        """Initialize temporal block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            dilation: Dilation factor.
            dropout: Dropout probability.
        """
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Output tensor of shape (batch, channels, time).
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample:
            residual = self.downsample(residual)

        out = out + residual
        out = self.relu_out(out)

        return out


class TCNBackbone(nn.Module):
    """Temporal Convolutional Network (TCN) backbone."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        """Initialize TCN backbone.

        Args:
            input_channels: Number of input feature channels.
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension size.
            num_layers: Number of temporal blocks.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
        super().__init__()

        layers: List[nn.Module] = []
        in_channels = input_channels

        for i in range(num_layers):
            out_channels = hidden_dim
            dilation = 2**i

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, time, features).

        Returns:
            Output logits of shape (batch, num_classes).
        """
        # Transpose to (batch, features, time) for Conv1d
        x = x.transpose(1, 2)

        # TCN layers
        x = self.network(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Classification
        x = self.fc(x)

        return x


class CNN1DBackbone(nn.Module):
    """1D CNN backbone for HAR."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """Initialize 1D CNN backbone.

        Args:
            input_channels: Number of input feature channels.
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension size.
            num_layers: Number of convolutional layers.
            dropout: Dropout probability.
        """
        super().__init__()

        layers: List[nn.Module] = []
        in_channels = input_channels

        for i in range(num_layers):
            out_channels = hidden_dim * (2**i)

            layers.extend(
                [
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=5, padding=2
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, time, features).

        Returns:
            Output logits of shape (batch, num_classes).
        """
        # Transpose to (batch, features, time) for Conv1d
        x = x.transpose(1, 2)

        # Convolutional layers
        x = self.conv_layers(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Classification
        x = self.fc(x)

        return x
