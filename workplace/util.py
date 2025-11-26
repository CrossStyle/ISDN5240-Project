# torch
import torch
import numpy as np
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
from curobo.geom.types import WorldConfig, Mesh, Cylinder, Cuboid
from robofab import ROBOFAB_DATA_DIR
import numpy as np
from scipy.spatial.transform import Rotation as R
from curobo.geom.types import Pose


tensor_args = TensorDeviceType()


def load_kuka_kin_model(gripper_width = 0.15):
    robot_name = "robot"
    config_file = load_yaml(ROBOFAB_DATA_DIR+ f"/robot/kuka/cfg.yaml")["robot_cfg"]
    config_kinematics = config_file['kinematics']
    config_kinematics["urdf_path"] = ROBOFAB_DATA_DIR + "/robot/kuka/" + config_kinematics["urdf_path"]
    config_kinematics["collision_spheres"] = ROBOFAB_DATA_DIR + "/robot/kuka/" + config_kinematics["collision_spheres"]
    config_kinematics["lock_joints"] = {"left_gripper_finger_joint": gripper_width, "right_gripper_finger_joint": gripper_width}

    robot_cfg = RobotConfig.from_dict(config_file, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    return config_file, kin_model


def rotate_euler(euler_angles, axis, angle, degrees=False):
    """
    Rotate given Euler angles by a specified angle around an arbitrary axis.

    Parameters:
    - euler_angles: tuple or list of 3 floats, Euler angles (roll, pitch, yaw) in radians (or degrees if degrees=True).
    - axis: tuple or list of 3 floats, the arbitrary axis vector (will be normalized).
    - angle: float, the rotation angle in radians (or degrees if degrees=True).
    - degrees: bool, if True, input angles are in degrees; otherwise, in radians.

    Returns:
    - new_euler: tuple of 3 floats, the new Euler angles after rotation, in the same units as input.
    """
    # Normalize the axis vector
    axis = np.array(axis)
    axis /= np.linalg.norm(axis)

    # Create rotation from Euler angles
    rot_euler = R.from_euler('xyz', euler_angles, degrees=degrees)

    # Create rotation from axis-angle
    rot_axis_angle = R.from_rotvec(axis * angle if not degrees else axis * np.deg2rad(angle))

    # Compose the rotations: apply the new rotation to the existing one
    # Note: Order matters; here we apply the new rotation after the original (right-multiply)
    # If you want to apply it before, use rot_axis_angle * rot_euler
    new_rot = rot_euler * rot_axis_angle

    # Convert back to Euler angles
    new_euler = new_rot.as_euler('xyz', degrees=degrees)

    return tuple(new_euler)


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def gen_fully_placed_world(scaffold_structure, valid_tube_index, world_cfg):
    tube_cylinder = []
    tube_mesh = []
    for tube_id, tube in scaffold_structure.items():
        if tube_id not in valid_tube_index:
            continue

        start_p = tube["from_p"]
        start_p[0] -= 0.2
        end_p = tube["to_p"]
        end_p[0] -= 0.2

        # get rotation quaternion from start_p to end_p
        direction = np.array(end_p) - np.array(start_p)
        direction = direction / np.linalg.norm(direction)
        # Assuming the cylinder's default orientation is along the z-axis
        z_axis = np.array([0, 0, 1])
        # Compute euler angles
        if np.allclose(direction, z_axis):
            roll, pitch, yaw = 0, 0, 0
        elif np.allclose(direction, -z_axis):
            roll, pitch, yaw = 0, np.pi, 0
        else:
            roll = np.arctan2(direction[0], direction[1])
            yaw = np.arccos(direction[2])
            pitch = 0
        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)

        cylinder = Cylinder(
            name=f"tube{tube_id}",
            radius=0.024,
            height=np.linalg.norm(np.array(end_p) - np.array(start_p)),
            # pose is [x, y, z, qx, qy, qz, qw]
            pose=[(start_p[0] + end_p[0]) / 2, (start_p[1] + end_p[1]) / 2, (start_p[2] + end_p[2]) / 2, qx, qy, qz,
                  qw],
            color=[0.7, 0.7, 0.7, 1.0],
        )
        tube_cylinder.append(cylinder)
        mesh = WorldConfig(cylinder=[cylinder]).get_mesh_world(merge_meshes=True).mesh[0]  # .get_trimesh_mesh()
        tube_mesh.append(mesh)

    fully_placed_world = WorldConfig(
        cylinder=[*world_cfg.cylinder] + tube_cylinder,
        cuboid=[*world_cfg.cuboid],
    )
    return fully_placed_world, tube_mesh, tube_cylinder


def update_frame(original_quat, new_pos, start_point, end_point):
    """
    Updates the frame to a new position and rotates it so that its z-axis aligns with the direction from start_point to end_point.

    Parameters:
    - original_pos: np.array (3,) - Original position (not used for new position, but included for completeness).
    - original_quat: np.array (4,) - Original quaternion in [x, y, z, w] format.
    - new_pos: np.array (3,) - New position for the frame.
    - start_point: np.array (3,) - Starting point for the z-axis direction.
    - end_point: np.array (3,) - Ending point for the z-axis direction.

    Returns:
    - new_pos: np.array (3,) - The new position.
    - new_quat: np.array (4,) - The updated quaternion in [x, y, z, w] format.
    """
    # Compute the target z-direction
    direction = end_point - start_point
    if np.linalg.norm(direction) == 0:
        raise ValueError("Start and end points cannot be the same; direction vector would be zero.")
    new_z = direction / np.linalg.norm(direction)

    # Get the current rotation and z-axis
    rot = R.from_quat(original_quat)
    current_z = rot.apply(np.array([0, 0, 1]))

    # If already aligned, no rotation needed
    if np.allclose(current_z, new_z, atol=1e-6):
        new_rot = rot
    else:
        # Compute the axis and angle for the alignment rotation
        cross_prod = np.cross(current_z, new_z)
        axis_norm = np.linalg.norm(cross_prod)
        if axis_norm == 0:
            # If vectors are opposite, choose an arbitrary perpendicular axis
            if np.dot(current_z, new_z) < 0:
                perp_vector = np.array([1, 0, 0]) if abs(current_z[0]) < 1e-6 else np.array([0, 1, 0])
                axis = np.cross(current_z, perp_vector)
                axis /= np.linalg.norm(axis)
                angle = np.pi
            else:
                angle = 0
                axis = np.array([1, 0, 0])  # Dummy, won't be used
        else:
            axis = cross_prod / axis_norm
            angle = np.arccos(np.clip(np.dot(current_z, new_z), -1.0, 1.0))

        # Create the alignment rotation
        align_rot = R.from_rotvec(angle * axis)

        # Apply to original rotation (compose: align_rot * rot)
        new_rot = align_rot * rot

    # Return new position and quaternion
    transform = np.eye(4)
    transform[:3, :3] = new_rot.as_matrix()
    transform[:3, 3] = new_pos
    return transform


def tube_info(scaffold_structure, valid_tube_index):
    tube_cylinder = []
    tube_mesh = []
    tube_pose = []
    tube_frames = []
    for tube_id, tube in scaffold_structure.items():
        if tube_id not in valid_tube_index:
            continue

        start_p = tube["from_p"]
        start_p[0] -= 0.2
        end_p = tube["to_p"]
        end_p[0] -= 0.2

        # get rotation quaternion from start_p to end_p
        direction = np.array(end_p) - np.array(start_p)
        direction = direction / np.linalg.norm(direction)
        # Assuming the cylinder's default orientation is along the z-axis
        z_axis = np.array([0, 0, 1])
        # Compute euler angles
        if np.allclose(direction, z_axis):
            roll, pitch, yaw = 0, 0, 0
        elif np.allclose(direction, -z_axis):
            roll, pitch, yaw = 0, np.pi, 0
        else:
            roll = np.arctan2(direction[0], direction[1])
            yaw = np.arccos(direction[2])
            pitch = 0
        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)

        cylinder = Cylinder(
            name=f"tube{tube_id}",
            radius=0.024,
            height=np.linalg.norm(np.array(end_p) - np.array(start_p)),
            # pose is [x, y, z, qx, qy, qz, qw]
            pose=[(start_p[0] + end_p[0]) / 2, (start_p[1] + end_p[1]) / 2, (start_p[2] + end_p[2]) / 2, qx, qy, qz,
                  qw],
            color=[0.7, 0.7, 0.7, 1.0],
        )
        tube_cylinder.append(cylinder)

        # mesh = trimesh.creation.cylinder(radius=0.024, height=np.linalg.norm(np.array(end_p)-np.array(start_p)), sections=16)
        # mesh.apply_translation([(start_p[0]+end_p[0])/2, (start_p[1]+end_p[1])/2, (start_p[2]+end_p[2])/2])
        mesh = WorldConfig(cylinder=[cylinder]).get_mesh_world(merge_meshes=True).mesh[0]  # .get_trimesh_mesh()
        tube_mesh.append(mesh)

        tube_pose_rotate = []
        tube_pose_rotate_pos = []
        tube_pose_rotate_quat = []
        for deg in range(0, 360, 5):
            # # get rotation matrix
            # rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
            # # # Rotation along direction's normal by deg degrees
            # rotation_matrix = R.from_rotvec(direction * np.deg2rad(deg)).as_matrix() @ rotation_matrix
            # r = R.from_matrix(rotation_matrix)
            # qx, qy, qz, qw = r.as_quat()
            [new_roll, new_pitch, new_yaw] = rotate_euler((roll, pitch, yaw), direction, np.deg2rad(deg))
            qx, qy, qz, qw = euler_to_quaternion(new_roll, new_pitch, new_yaw)

            tube_pose_rotate_pos.append(
                [(start_p[0] + end_p[0]) / 2, (start_p[1] + end_p[1]) / 2, (start_p[2] + end_p[2]) / 2])
            tube_pose_rotate_quat.append([qx, qy, qz, qw])

            tube_pose_rotate = Pose(
                position=torch.tensor(tube_pose_rotate_pos, **tensor_args.as_torch_dict()),
                quaternion=torch.tensor(tube_pose_rotate_quat, **tensor_args.as_torch_dict())
            )

        tube_pose.append(tube_pose_rotate)
    return tube_cylinder, tube_mesh, tube_pose


def tube_frame(scaffold_structure, valid_tube_index):
    frames = []
    for tube_id, tube in scaffold_structure.items():
        if tube_id not in valid_tube_index:
            continue

        start_p = np.array(tube["from_p"])
        start_p[0] -= 0.2
        end_p = np.array(tube["to_p"])
        end_p[0] -= 0.2

        new_pos = 0.5 * (end_p + start_p)
        frame= update_frame([0, 0, 0, 1], new_pos, start_p, end_p)
        frames.append(frame)
    return frames


def generate_rotated_frames(original_transform, angles_degrees=None, num_steps=72):
    """
    Generates a list of new 4x4 transformation matrices by rotating the original transform around its local z-axis.

    Parameters:
    - original_transform: np.array (4,4) - The original 4x4 transformation matrix.
    - angles_degrees: list or np.array - Specific angles in degrees to rotate by. If None, uses linspace from 0 to 360.
    - num_steps: int - Number of steps if angles_degrees is None (includes 0, excludes 360 if full circle).

    Returns:
    - rotated_transforms: list of np.array (4,4) - List of rotated transformation matrices.
    """
    if angles_degrees is None:
        angles_degrees = np.linspace(0, 360, num_steps + 1)[:-1]  # 0 to almost 360, avoiding duplicate

    R_original = original_transform[:3, :3]
    t = original_transform[:3, 3]

    rotated_transforms = []
    pos = []
    rot = []
    for theta_deg in angles_degrees:
        theta_rad = np.deg2rad(theta_deg)
        # Rotation matrix around z-axis
        R_z = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad), 0],
            [np.sin(theta_rad), np.cos(theta_rad), 0],
            [0, 0, 1]
        ])
        # Local rotation: post-multiply (right-multiply) the z-rotation
        new_R = R_original @ R_z
        new_transform = np.eye(4)
        new_transform[:3, :3] = new_R
        new_transform[:3, 3] = t
        rotated_transforms.append(new_transform[np.newaxis, :, :])
        rot.append(new_R[np.newaxis, :, :])
        # quat.append(R.from_matrix(new_R).as_quat()[np.newaxis, :, :])
        pos.append(t[np.newaxis, :])

    rot = torch.tensor(np.vstack(rot), **tensor_args.as_torch_dict())
    pos = torch.tensor(np.vstack(pos), **tensor_args.as_torch_dict())

    poses = Pose(position=pos, rotation=rot)
    return np.vstack(rotated_transforms), poses