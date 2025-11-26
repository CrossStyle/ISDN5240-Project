import numpy as np
import torch
import trimesh.util
from curobo.geom.types import WorldConfig, Pose, Mesh
from curobo.types.state import JointState
from trimesh import Trimesh
from trimesh.voxel import VoxelGrid
from trimesh.creation import box
from curobo.geom.transform import matrix_to_quaternion, quaternion_to_matrix
from curobo.types.base import TensorDeviceType
from typing import List
from robofab import ROBOFAB_DATA_DIR

def from_voxels_to_mesh(voxels: np.ndarray, origin: np.ndarray, voxel_size=0.025):
    voxel_meshes = []
    points = []
    for id in range(voxels.shape[0]):
        xcoord = (voxels[id, 0] + 0.5) * voxel_size + origin[0]
        ycoord = (voxels[id, 1] + 0.5) * voxel_size + origin[1]
        zcoord = (voxels[id, 2] + 0.5) * voxel_size + origin[2]
        offset = np.array([xcoord, ycoord, zcoord], dtype=float)
        voxel_mesh = box(extents=(voxel_size, voxel_size, voxel_size))
        voxel_mesh.apply_translation(offset)
        voxel_meshes.append(voxel_mesh)

        shrink_voxel_mesh = box(extents=(voxel_size * 0.9, voxel_size * 0.9, voxel_size * 0.9))
        shrink_voxel_mesh.apply_translation(offset)
        points.append(shrink_voxel_mesh.vertices)
    return trimesh.util.concatenate(voxel_meshes), np.vstack(points)


def compute_part_transformation_from_place_to_pick(part: Trimesh,
                                                   pick_station,
                                                   voxel_size):
    import polyscope as ps

    # enumerate different orientation
    tensor_args = TensorDeviceType()
    zaxes = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                         **tensor_args.as_torch_dict())
    yaxes = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                         **tensor_args.as_torch_dict())
    zaxes = zaxes.repeat_interleave(6, dim=0)
    yaxes = torch.tile(yaxes, (6, 1))
    xaxes = torch.cross(yaxes, zaxes, dim=1)
    flag = torch.linalg.norm(xaxes, dim=1) > 0.5
    xaxes = xaxes[flag]
    yaxes = yaxes[flag]
    zaxes = zaxes[flag]
    mats = torch.zeros((xaxes.shape[0], 3, 3), **tensor_args.as_torch_dict())
    mats[:, :, 0] = xaxes
    mats[:, :, 1] = yaxes
    mats[:, :, 2] = zaxes
    mats = mats.cpu().numpy()

    # check transformation
    transformations = []
    _, pick_parts, pick_parts_points, _ = pick_station

    for pick_part_id, pick_part in enumerate(pick_parts):
        points = pick_parts_points[pick_part_id].copy()
        points = points - pick_part.center_mass
        if abs(pick_part.volume - part.volume) > voxel_size * voxel_size * voxel_size / 2:
            continue
        for mat_id in range(mats.shape[0]):
            mat4x4 = np.eye(4)
            mat4x4[:3, :3] = mats[mat_id, :, :]
            mat4x4[:3, 3] = part.center_mass
            mat4x4 = np.linalg.inv(mat4x4)
            new_part = Trimesh(part.vertices, part.faces)
            new_part = new_part.apply_transform(mat4x4)
            contains = new_part.contains(points)
            if contains.all():
                mat4x4[:3, 3] += pick_part.center_mass
                transformations.append(mat4x4.copy())

    return transformations


def create_robot_pick_station(robot_id: int = 0,
                              voxel_size: float = 0.025,
                              offset: List[float] = [0, 0, 0],
                              shapes=["S-", "S+", "S2-", "T-", "T+", "T+2", "O", "L4-", "L4+", "L4|-", "L4|+", "I4"]):
    if robot_id == 0:
        offset0 = np.array([0.279553 + offset[0],
                            -0.446842  + offset[1],
                            -0.01 + offset[2]]) #cali50

    if robot_id == 1:
        offset0 = np.array([ 1.08559515  + offset[0], 0.39972298 + offset[1], -0.01063572 + offset[2]])
    pick_stations = []
    for shape in shapes:
        pick_station = create_pick_station(robot_id=robot_id, offset=offset0, voxel_size=voxel_size, shape=shape)
        pick_stations.append(pick_station)
    return pick_stations

def ground_mesh():
    obstacle = Mesh(
        name=f"ground",
        pose=[0, 0, 0, 1, 0, 0, 0],
        file_path=f"{ROBOFAB_DATA_DIR}/world/ground_with_platform.obj",
        scale = (1E-3, 1E-3, 1E-3)
    )
    return obstacle

def get_pick_station_world_config(obj_file_path):
    obstacle = Mesh(
        name=f"pick_station",
        pose=[0, 0, 0, 1, 0, 0, 0],
        file_path=obj_file_path,
        scale=(1, 1, 1),
    )
    return WorldConfig(mesh=[obstacle, ground_mesh()])

def create_pick_station(robot_id,
                        offset: np.ndarray,
                        voxel_size=0.025,
                        shape="S"):
    pick_parts = []
    pick_parts_points = []

    if shape == "S-":
        # flat s
        voxels = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "S+":
        # flat s
        voxels = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [2, 0, 1]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "S2-":
        # vertical s
        voxels = np.array([[0, 0, 0], [1, 0, 0], [1, -1, 0], [2, -1, 0]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "L4-":
        # left L
        voxels = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "L4|-":
        # left vertical L
        voxels = np.array([[1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "L4+":
        # right L
        voxels = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 1, 0]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "L4|+":
        # left vertical L
        voxels = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 0, 1]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "O":
        voxels = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "I4":
        voxels = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "T-":
        # flat T
        voxels = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "T+":
        # vertical T
        voxels = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 1]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    if shape == "T+2":
        # vertical T 2
        voxels = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 1], [1, 0, 1]])
        voxel_mesh, points = from_voxels_to_mesh(voxels, offset, voxel_size)
        pick_parts.append(voxel_mesh)
        pick_parts_points.append(points)

    pick_station_mesh = trimesh.util.concatenate(pick_parts)
    filename = ROBOFAB_DATA_DIR + f"/pick_station/{robot_id}_{shape}.obj"
    pick_station_mesh.export(filename)
    pick_station_world_config = get_pick_station_world_config(filename)

    return shape, pick_parts, pick_parts_points, pick_station_world_config
