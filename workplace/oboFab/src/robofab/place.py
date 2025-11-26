from typing import List
import trimesh
import numpy as np

from curobo.types.base import TensorDeviceType
from curobo.geom.transform import pose_to_matrix, matrix_to_quaternion
from curobo.geom.types import Pose
import torch


def poses_to_frames(poses: Pose):
    mat4x4 = pose_to_matrix(position=poses.position, quaternion=poses.quaternion)
    return mat4x4.cpu().numpy()

def compute_part_place_pose(part: trimesh.Trimesh,
                            tool_offset: float = 0.1,
                            zaxis: np.ndarray = None,
                            tensor_args = TensorDeviceType()):
    
    # voxel center
    tensor_args = TensorDeviceType()
    grid_size = part.bounds[1, 1] - part.bounds[0, 1]
    nx = round((part.bounds[1, 0] - part.bounds[0, 0]) / grid_size)
    nz = round((part.bounds[1, 2] - part.bounds[0, 2]) / grid_size)
    ycoord = (part.bounds[1, 1] + part.bounds[0, 1]) / 2
    points = []
    center_of_mass = part.center_mass
    for id in range(0, nx):
        for jd in range(0, nz):
            xcoord = part.bounds[0, 0] + (id + 0.5) * grid_size
            zcoord = part.bounds[0, 2] + (jd + 0.5) * grid_size
            point = [xcoord, ycoord, zcoord]
            if np.all(np.abs(point - center_of_mass) < grid_size * 1.25):
                points.append(point)
    points = np.array(points)
    points = points[part.contains(points)]

    # quad
    zaxes = []
    yaxes = []

    if zaxis is None:
        yaxes.extend([[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]])
        zaxes.extend([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]])
        zaxes.extend([[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]])
        yaxes.extend([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]])
    else:
        yaxes = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        zaxes = [zaxis.tolist()] * 6
    zaxes = torch.tensor(zaxes, **tensor_args.as_torch_dict())
    yaxes = torch.tensor(yaxes, **tensor_args.as_torch_dict())
    xaxes = torch.cross(yaxes, zaxes, dim=1)

    # remove zero
    flag = torch.linalg.norm(xaxes, dim = 1) > 1E-6
    xaxes = xaxes[flag, :]
    yaxes = yaxes[flag, :]
    zaxes = zaxes[flag, :]

    mat = torch.zeros((zaxes.shape[0], 3, 3), **tensor_args.as_torch_dict())
    mat[:, :, 0] = xaxes
    mat[:, :, 1] = yaxes
    mat[:, :, 2] = zaxes
    quat = matrix_to_quaternion(mat)
    quat = torch.tile(quat, (points.shape[0], 1))

    # origin
    origin = torch.tensor(points, **tensor_args.as_torch_dict())
    origin = torch.repeat_interleave(origin, zaxes.shape[0], dim=0)
    zaxes = torch.tile(zaxes, (points.shape[0], 1))
    origin = origin - zaxes * tool_offset
    poses = Pose(position=origin, quaternion=quat)
    return poses
