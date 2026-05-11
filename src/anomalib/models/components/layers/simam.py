# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SimAM attention layer.

This module implements the parameter-free SimAM attention mechanism from the paper:
"SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks"
(https://proceedings.mlr.press/v139/yang21o.html)
"""

import torch
from torch import nn


class SimAM(nn.Module):
    """Apply parameter-free attention to 4D feature maps.

    Args:
        lambda_value (float, optional): Stabilizer term used in the energy
            function. Defaults to ``1e-4``.
    """

    def __init__(self, lambda_value: float = 1e-4) -> None:
        super().__init__()
        self.lambda_value = lambda_value
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply SimAM attention.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor: Attended tensor of same shape as input.
        """
        _, _, height, width = inputs.shape
        num_elements = height * width - 1

        if num_elements <= 0:
            return inputs

        centered = (inputs - inputs.mean(dim=(2, 3), keepdim=True)).pow(2)
        norm = 4 * (centered.sum(dim=(2, 3), keepdim=True) / num_elements + self.lambda_value)
        attention = centered / norm + 0.5
        return inputs * self.activation(attention)
