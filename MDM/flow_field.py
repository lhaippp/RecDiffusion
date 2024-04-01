from typing import Tuple
import torch

def get_Hs(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Get Homography matrix from source to target
    :param source: (N, 4, 2) 4 points (x, y) Tensor
    :param target: (N, 4, 2) 4 points (x, y) Tensor
    :return: (N, 3, 3) Homography matrix
    """
    # Generate A
    N = source.shape[0]
    assert source.shape == (N, 4, 2) and target.shape == (N, 4, 2)
    A = torch.zeros(N, 8, 9)
    ones = torch.ones((N,)).to(source.device)
    zeros = torch.zeros((N,)).to(source.device)
    for i in range(4):
        x, y, u, v = source[:, i, 0], source[:, i, 1], target[:, i, 0], target[:, i, 1]
        A[:, 2*i, :] = torch.stack([-x, -y, -ones, zeros, zeros, zeros, u*x, u*y, u], 1)
        A[:, 2*i+1, :] = torch.stack([zeros, zeros, zeros, -x, -y, -ones, v*x, v*y, v], 1)
    # Get H from V
    _, _, Vh = torch.linalg.svd(A)
    H = Vh[:, -1].reshape(-1, 3, 3)
    # return H[:] / H[:, 2, 2]
    H = H / H[:, 2, 2].reshape(-1, 1, 1)
    return H

def get_uniform_grid(grid_count: Tuple[int, int], size: Tuple[int, int]) -> torch.Tensor:
    """
    Get uniform grid
    :param grid_count: (cx, cy) Tuple grid count
    :param size: (w, h) Tuple image size
    :return: (cx + 1, cy + 1, 2) Tensor grid
    """
    grid_cx, grid_cy = grid_count
    grid_h, grid_w = size[0] / grid_cx, size[1] / grid_cy
    grid = torch.zeros(grid_cx + 1, grid_cy + 1, 2)
    for i in range(grid_cx + 1):
        for j in range(grid_cy + 1):
            grid[i, j] = torch.Tensor([i * grid_h, j * grid_w])
    return grid

def calculate_flow_field(theta: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Calculate flow field from theta
    :param theta: (B, cx + 1, cy + 1, 2) Tensor
    :param size: (2,) Tuple output image size
    :return: (B, size[0], size[1], 2) Tensor flow field
    """
    if len(theta.shape) == 3:
        theta = theta.unsqueeze(0)
    batch_size = theta.shape[0]
    grid_cx, grid_cy = theta.shape[1], theta.shape[2]
    grid_cx, grid_cy = grid_cx - 1, grid_cy - 1
    grid_h, grid_w = 2 / grid_cx, 2 / grid_cy

    Hs = torch.zeros(batch_size, grid_cx, grid_cy, 9).to(theta.device)
    for i in range(grid_cx):
        for j in range(grid_cy):
            grid_x, grid_y = i * grid_h - 1, j * grid_w - 1
            source = torch.Tensor([[
                [grid_x, grid_y],
                [grid_x + grid_h, grid_y],
                [grid_x + grid_h, grid_y + grid_w],
                [grid_x, grid_y + grid_w]]]).repeat(batch_size, 1, 1).to(theta.device)
            target = torch.cat([
                theta[:, i, j].unsqueeze(1),
                theta[:, i + 1, j].unsqueeze(1),
                theta[:, i + 1, j + 1].unsqueeze(1),
                theta[:, i, j + 1].unsqueeze(1)], dim=1)
            H = get_Hs(source, target)
            Hs[:, i, j] = H.reshape(-1, 9)
    
    # Copy Hs to full size
    # (B, cx, cy, 9) -> (B, 9, cx, cy) -> (B, 9, size[0], size[1]) -> (B, size[0], size[1], 9) -> (B, size[0] * size[1], 3, 3)
    Hs = Hs.permute(0, 3, 1, 2)
    Hs = torch.nn.functional.interpolate(Hs, size=size, mode='nearest')
    Hs = Hs.permute(0, 2, 3, 1).reshape(batch_size, size[0] * size[1], 3, 3)

    # generate mesh (0-1)
    grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, size[0]), torch.linspace(-1, 1, size[1]))
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    grid_z = torch.ones((size[0] * size[1],))

    # calculate each dimension of the output
    grid = torch.stack([grid_x, grid_y, grid_z], 1).to(theta.device)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
    grid = torch.matmul(Hs, grid.unsqueeze(-1)).reshape(batch_size, size[0], size[1], 3)
    grid = grid[:, :, :, :2] / grid[:, :, :, 2:]
    return grid

def convert_flow_field_to_uv(flow_field: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, image_shape[0], flow_field.shape[-3]), torch.linspace(0, image_shape[1], flow_field.shape[-2]))
    flow_field[:, :, :, 0] -= grid_y.to(flow_field.device)
    flow_field[:, :, :, 1] -= grid_x.to(flow_field.device)
    return flow_field

def convert_mesh_to_dxdy(mesh_wo_batch_dim: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    uv = convert_flow_field_to_uv(calculate_flow_field(mesh_wo_batch_dim, image_size), image_size).squeeze(0)
    # uv[:, :, 0] *= image_size[1]
    # uv[:, :, 1] *= image_size[0]
    return uv.permute((2, 0, 1))