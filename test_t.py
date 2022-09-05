import math
import torch
x = torch.Tensor([1,1])
phi = torch.tensor(math.pi/ 4)
s = torch.sin(phi)
c = torch.cos(phi)
rot = torch.stack([torch.stack([c, -s]),
                   torch.stack([s, c])])
x_rot = x @ rot.t() 
print(x_rot)
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeometry.utils import create_meshgrid
from torchgeometry.core.transformations import transform_points

from math import pi

__all__ = [
    "HomographyWarper",
    "homography_warp",
]
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from torchgeometry.core.conversions import deg2rad
from torchgeometry.core.homography_warper import homography_warp

# layer api


def get_rotation_matrix2d(center, angle, scale):
    r"""Calculates an affine matrix of 2D rotation.
    The function calculates the following matrix:
    .. math::
        \begin{bmatrix}
            \alpha & \beta & (1 - \alpha) \cdot \text{x}
            - \beta \cdot \text{y} \\
            -\beta & \alpha & \beta \cdot \text{x}
            + (1 - \alpha) \cdot \text{y}
        \end{bmatrix}
    where
    .. math::
        \alpha = \text{scale} \cdot cos(\text{angle}) \\
        \beta = \text{scale} \cdot sin(\text{angle})
    The transformation maps the rotation center to itself
    If this is not the target, adjust the shift.
    Args:
        center (Tensor): center of the rotation in the source image.
        angle (Tensor): rotation angle in degrees. Positive values mean
            counter-clockwise rotation (the coordinate origin is assumed to
            be the top-left corner).
        scale (Tensor): isotropic scale factor.
    Returns:
        Tensor: the affine matrix of 2D rotation.
    Shape:
        - Input: :math:`(B, 2)`, :math:`(B)` and :math:`(B)`
        - Output: :math:`(B, 2, 3)`
    Example:
        >>> center = torch.zeros(1, 2)
        >>> scale = torch.ones(1)
        >>> angle = 45. * torch.ones(1)
        >>> M = tgm.get_rotation_matrix2d(center, angle, scale)
        tensor([[[ 0.7071,  0.7071,  0.0000],
                 [-0.7071,  0.7071,  0.0000]]])
    """
    if not torch.is_tensor(center):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}"
                        .format(type(angle)))
    if not torch.is_tensor(scale):
        raise TypeError("Input scale type is not a torch.Tensor. Got {}"
                        .format(type(scale)))
    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError("Input center must be a Bx2 tensor. Got {}"
                         .format(center.shape))
    if not len(angle.shape) == 1:
        raise ValueError("Input angle must be a B tensor. Got {}"
                         .format(angle.shape))
    if not len(scale.shape) == 1:
        raise ValueError("Input scale must be a B tensor. Got {}"
                         .format(scale.shape))
    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Got {}"
                         .format(center.shape, angle.shape, scale.shape))
    # convert angle and apply scale
    angle_rad = deg2rad(angle)
    alpha = torch.cos(angle_rad) * scale
    beta = torch.sin(angle_rad) * scale

    # unpack the center to x, y coordinates
    x, y = center[..., 0], center[..., 1]

    # create output tensor
    batch_size, _ = center.shape
    M = torch.zeros(batch_size, 2, 3, device=center.device, dtype=center.dtype)
    M[..., 0, 0] = alpha
    M[..., 0, 1] = beta
    M[..., 0, 2] = (1. - alpha) * x - beta * y
    M[..., 1, 0] = -beta
    M[..., 1, 1] = alpha
    M[..., 1, 2] = beta * x + (1. - alpha) * y
    return M



B=2
center=torch.stack([torch.Tensor([0,0]) for _ in range(B)])
input=torch.stack([torch.Tensor([1,0]) for _ in range(B)])
scale=torch.ones(B)
angle=-90 * torch.ones(B)
M=get_rotation_matrix2d(center,angle,scale)
print(input)
print(M.shape)
print(input@M)