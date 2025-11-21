import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation


def sample_sphere(radius, num_points=1000):
    """ Sample points uniformly on a sphere surface using Fibonacci spiral approximation. """
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    points = np.stack([x, y, z], axis=1)
    return radius * points


def sample_box(size, num_points_per_face=100):
    """ Sample points on the surface of a box (axis-aligned for simplicity). size: array-like [half_x, half_y, half_z] """
    size = np.array(size)
    points = []
    # Sample each of the 6 faces with a grid
    for dim in range(3):
        for sign in [-1, 1]:
            # Fixed coordinate for this face
            fixed = sign * size[dim]
            # Grid on the other two dimensions
            u = np.linspace(-size[(dim + 1) % 3], size[(dim + 1) % 3], int(np.sqrt(num_points_per_face)))
            v = np.linspace(-size[(dim + 2) % 3], size[(dim + 2) % 3], int(np.sqrt(num_points_per_face)))
            uu, vv = np.meshgrid(u, v)
            face_points = np.zeros((uu.size, 3))
            face_points[:, dim] = fixed
            face_points[:, (dim + 1) % 3] = uu.ravel()
            face_points[:, (dim + 2) % 3] = vv.ravel()
            points.append(face_points)
    return np.vstack(points)


def sample_cylinder(radius, height, num_points=1000):
    """ Sample points on a cylinder surface (sides + caps), axis along z. height: full height (MuJoCo size[1] * 2) """
    points = []
    # Side surface
    num_side = int(num_points * 0.8)
    theta = np.random.uniform(0, 2 * np.pi, num_side)
    z = np.random.uniform(-height / 2, height / 2, num_side)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points.append(np.stack([x, y, z], axis=1))
    # Top and bottom caps (disks)
    num_cap = int(num_points * 0.1)
    for z_val in [-height / 2, height / 2]:
        r = np.sqrt(np.random.uniform(0, 1, num_cap)) * radius
        theta = np.random.uniform(0, 2 * np.pi, num_cap)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full(num_cap, z_val)
        points.append(np.stack([x, y, z], axis=1))
    return np.vstack(points)


def apply_pose(points, pos, quat):
    rot = Rotation.from_quat(quat).as_matrix()
    return (rot @ points.T).T + pos


# Define geoms from XML
geoms = [
    {
        'name': 'wall',
        'type': 'box',
        'size': [3, 0.025, 1.5],
        'pos': [3.2, 0, 1.5],
        'euler': [0, 0, 1.57]
    },
    {
        'name': 'wall2',
        'type': 'cylinder',
        'size': [0.5, 1.5],
        'pos': [3.2, -1.8, 1.5],
        'euler': [0, 0, 1.57]
    },
    {
        'name': 'wall3',
        'type': 'cylinder',
        'size': [0.5, 1.5],
        'pos': [3.2, 1.8, 1.5],
        'euler': [0, 0, 1.57]
    }
]

# Generate point clouds
point_clouds = {}
for geom in geoms:
    euler = geom['euler']
    pos = np.array(geom['pos'])
    quat = Rotation.from_euler('xyz', euler).as_quat()  # Assumes euler in radians, 'xyz' sequence (MuJoCo default)

    if geom['type'] == 'box':
        points_local = sample_box(geom['size'], num_points_per_face=2000)
    elif geom['type'] == 'cylinder':
        radius, half_height = geom['size']
        height = 2 * half_height
        points_local = sample_cylinder(radius, height, num_points=1000)
    else:
        raise ValueError(f"Unsupported geom type: {geom['type']}")

    points_world = apply_pose(points_local, pos, quat)
    point_clouds[geom['name']] = points_world

all_points = np.vstack(list(point_clouds.values()))
half_points = all_points[all_points[:, 0] <= 3.2]

proj_xy = half_points[:, :2]  # (N, 2)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(half_points)
pcd.paint_uniform_color([0.8, 0.8, 0.4])
o3d.visualization.draw_geometries([pcd])

# save
# import os
# ply_path = os.path.join('data', "mujoco_geoms_pointcloud.ply")
# o3d.io.write_point_cloud(ply_path, pcd)
#
# npy_proj_path = os.path.join('data', "proj_xy.npy")
# np.save(npy_proj_path, proj_xy)


