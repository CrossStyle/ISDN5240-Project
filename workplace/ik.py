import torch
from robofab.place import poses_to_frames
from curobo.types.robot import RobotConfig
from curobo.geom.types import Pose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.types import WorldConfig, Mesh, Cylinder, Cuboid
from curobo.geom.types import Pose
import trimesh
from util import load_kuka_kin_model
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.util.trajectory import InterpolateType
import numpy as np
from curobo.geom.transform import matrix_to_quaternion


tensor_args = TensorDeviceType()


def get_kuka_ik_solver(gripper_width=0.15, world_config: WorldConfig = None, num_seeds=500):

    config_file, _ = load_kuka_kin_model(gripper_width)
    robot_cfg = RobotConfig.from_dict(config_file)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_model=world_config,
        tensor_args=tensor_args,
        use_cuda_graph=True,
        rotation_threshold=0.001,
        position_threshold=0.001,
        num_seeds=num_seeds,
        self_collision_check=False,
        self_collision_opt=False,
        collision_activation_distance=0.02
    )
    ik_solver = IKSolver(ik_config)
    return ik_solver


### initialize Plan IK function
def get_poses_by_flag(poses: Pose, flag: torch.tensor):
    return Pose(position=poses.position[flag, :], quaternion=poses.quaternion[flag, :])


def plan_ik(poses: Pose, world_config: WorldConfig, gripper_width: float = 0.15):
    ik_solver = get_kuka_ik_solver(gripper_width=gripper_width, world_config=world_config)
    result = ik_solver.solve_batch(poses)
    ik_solver.reset_cuda_graph()
    if result.success.any():
        flag = result.success
        success_qs = result.solution[result.success]
        sucess_pose = get_poses_by_flag(poses, flag.flatten())
        return (success_qs, sucess_pose, flag.flatten())
    else:
        return None


def plan_ik_seed_config(poses: Pose, world_config: WorldConfig, seed_config: JointState,
                        gripper_width: float = 0.15):
    ik_solver = get_kuka_ik_solver(gripper_width=gripper_width, world_config=world_config, num_seeds=1)
    seed_config = seed_config.position.view(1, -1)
    result = ik_solver.solve_batch(goal_pose=poses, seed_config=seed_config, num_seeds=1)
    if result.success.any():
        flag = result.success
        success_qs = result.solution[result.success]
        sucess_pose = get_poses_by_flag(poses, flag.flatten())
        return (success_qs, sucess_pose, flag.flatten())
    else:
        return None



def get_kuka_motion_gen(gripper_width=0.15, world_config: WorldConfig = None, num_seeds=1000):
    # config_file, _ = load_franka_kin_model(gripper_width)
    config_file, _ = load_kuka_kin_model(gripper_width)
    robot_cfg = RobotConfig.from_dict(config_file)
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_model=world_config,
        tensor_args=tensor_args,
        use_cuda_graph=True,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_ik_seeds=num_seeds,
        num_trajopt_seeds=num_seeds,
        interpolation_dt=0.01,
        interpolation_steps=1000,
        interpolation_type=InterpolateType.CUBIC,
        high_precision=False,
        self_collision_check=False,
        self_collision_opt=False,
        minimize_jerk=False,
        collision_activation_distance=0.02)
    return MotionGen(motion_gen_config)



def plan_motion_attached_object(start_state: JointState,
                                goal_poses: Pose,
                                attached_object: Mesh,
                                attached_state: JointState,
                                world_config: WorldConfig,
                                gripper_width: float = 0.15):
    motion_gen = get_kuka_motion_gen(world_config=world_config, gripper_width=gripper_width)
    place_poses = Pose(position=goal_poses.position.view(1, -1, 3), quaternion=goal_poses.quaternion.view(1, -1, 4))
    start_state.joint_names = motion_gen.kinematics.joint_names.copy()

    attached_state.joint_names = motion_gen.kinematics.joint_names.copy()
    motion_gen.attach_external_objects_to_robot(attached_state,
                                                [attached_object],
                                                surface_sphere_radius=0.015,
                                                link_name="attached_object")

    result = motion_gen.plan_goalset(start_state, place_poses, MotionGenPlanConfig(max_attempts=100))
    if result.success.any():
        qtraj = result.get_interpolated_plan().position
        return (qtraj, result.goalset_index.item())
    else:
        return None


def plan_motion_js_attached_object(start_state: JointState,
                                   goal_state: JointState,
                                   attached_object: Mesh,
                                   attached_state: JointState,
                                   world_config: WorldConfig,
                                   gripper_width: float = 0.15):
    motion_gen = get_kuka_motion_gen(world_config=world_config, gripper_width=gripper_width)
    start_state.joint_names = motion_gen.kinematics.joint_names.copy()
    goal_state.joint_names = motion_gen.kinematics.joint_names.copy()

    attached_state.joint_names = motion_gen.kinematics.joint_names.copy()
    motion_gen.attach_external_objects_to_robot(attached_state,
                                                [attached_object],
                                                surface_sphere_radius=0.015,
                                                link_name="attached_object")

    result = motion_gen.plan_single_js(start_state, goal_state, MotionGenPlanConfig(max_attempts=100))
    if result.success.any():
        qtraj = result.get_interpolated_plan().position
        return qtraj
    else:
        return None


# def compute_pick_place(tube_id, tube_pose, tube_cylinder, place_world, robot_kin_model):
#     ## 2.1 Compute place poses
#     place_result = plan_ik(tube_pose, place_world, 0.15)
#     if place_result is None:
#         raise RuntimeError("No solution found!")
#
#     # update
#     place_poses = place_result[1]
#
#     ## 2.3 Plan Place IK
#     ### Plan place approach IK
#     place_approach_poses = place_poses.clone()
#     place_approach_poses.position += torch.tensor([0, 0, 0.1], **tensor_args.as_torch_dict())
#     place_approach_result = plan_ik(place_approach_poses, place_world, 0.15)
#     if place_approach_result is None:
#         raise RuntimeError("No solution found!")
#
#     # update
#     place_approach_poses = place_approach_result[1]
#     place_poses = get_poses_by_flag(place_poses, place_approach_result[2])
#     # print("Number of place poses", place_poses.position.shape[0])
#
#     ### Compute place to pick transformation matrix
#     pick_pose_pos = [-1.5, -1.5, 0]
#     pick_pose_quat = [0, 1, -1, 0]
#     pick_poses = Pose(position=torch.tensor([pick_pose_pos] * len(place_poses), **tensor_args.as_torch_dict()),
#                       quaternion=torch.tensor([pick_pose_quat] * len(place_poses), **tensor_args.as_torch_dict()))
#     pick_frames = poses_to_frames(pick_poses)
#
#     ### Compute pick poses
#     pick_frames = torch.tensor(pick_frames, **tensor_args.as_torch_dict())
#     pick_poses = Pose(position=pick_frames[:, :3, 3], quaternion=matrix_to_quaternion(pick_frames[:, :3, :3]))
#
#     ### Plan pick ik
#     pick_result = plan_ik(pick_poses, place_world, 0.15)
#     if pick_result is None:
#         raise RuntimeError("No solution found!")
#
#     # update
#     pick_poses = pick_result[1]
#     place_poses = get_poses_by_flag(place_poses, pick_result[2])
#     place_approach_poses = get_poses_by_flag(place_approach_poses, pick_result[2])
#
#     ### Plan pick approach ik
#     pick_approach_poses = pick_poses.clone()
#     pick_approach_poses.position += torch.tensor([0, 0, 0.1], **tensor_args.as_torch_dict())
#     pick_approach_result = plan_ik(pick_approach_poses, place_world, 0.15)
#     if pick_approach_result is None:
#         raise RuntimeError("No solution found!")
#
#     # update
#     pick_approach_poses = pick_approach_result[1]
#     pick_poses = get_poses_by_flag(pick_poses, pick_approach_result[2])
#     place_poses = get_poses_by_flag(place_poses, pick_approach_result[2])
#     # print("Number of pick poses", place_poses.position.shape[0])
#
#     ## 4.1 Retract -> pick_approach
#     ### Choose one pick-place-approach pair
#
#     pick_place_id = 0
#     pick_pose = pick_poses.clone()[pick_place_id]
#     place_pose = place_poses.clone()[pick_place_id]
#     pick_approach_pose = pick_approach_poses.clone()[pick_place_id]
#     place_approach_pose = place_approach_poses.clone()[pick_place_id]
#
#     ### Compute attach objects and states
#
#     # mesh = trimesh.creation.cylinder(radius=0.024, height=np.linalg.norm(np.array(end_p) - np.array(start_p)),
#     #                                  sections=16)
#     # mesh.apply_translation(pick_pose_pos)
#     # mesh = Mesh(
#     #     name=f"tube_mesh{tube_id}",
#     #     pose=[*pick_pose_pos, *pick_pose_quat],
#     #     vertices=mesh.vertices,
#     #     faces=mesh.faces,
#     #     color=[0.7, 0.7, 0.7, 1.0],
#     # )
#
#     # pick_result = plan_ik(pick_pose, pick_world[3], 0.15)
#     pick_result = plan_ik(pick_pose, place_world, 0.15)
#     if pick_result is None:
#         raise RuntimeError("No solution found!")
#     # pick_q = pick_result[0].cpu().numpy().reshape(1, -1)
#
#     attached_state = JointState.from_position(pick_result[0])
#     mesh = trimesh.creation.cylinder(radius=0.024, height=tube_cylinder.height, sections=16)
#     mesh.apply_translation(pick_pose_pos)
#
#     tube_mesh = Mesh(
#         name=f"tube_mesh{tube_id}",
#         pose=[*pick_pose_pos, *pick_pose_quat],
#         vertices=mesh.vertices,
#         faces=mesh.faces,
#         color=[0.7,0.7,0.7,1.0],
#     )
#
#     attached_object = tube_mesh
#     start_state = JointState.from_position(robot_kin_model.retract_config.view(1, -1),
#                                            joint_names=robot_kin_model.joint_names)
#
#     attached_object_for_rendering = trimesh.Trimesh(attached_object.vertices.copy(), attached_object.faces.copy())
#     T = poses_to_frames(pick_pose).reshape(4, 4)
#     invT = np.linalg.inv(T)
#     attached_object_for_rendering.apply_transform(invT)
#
#     ### Plan motion
#     trajs_retract_pick_approach_result = plan_motion_attached_object(start_state, pick_approach_pose, attached_object,
#                                                                      attached_state, place_world, 0.15)
#     if trajs_retract_pick_approach_result is None:
#         raise RuntimeError("No solution found!")
#     # trajs_retract_pick = trajs_retract_pick_approach_result[0].clone().cpu().numpy()
#     # trajs_retract_pick = np.vstack([trajs_retract_pick, pick_q])
#
#     ### Resolve IK Jumping
#     pick_approach_state = JointState.from_position(trajs_retract_pick_approach_result[0][-1],
#                                                    joint_names=robot_kin_model.joint_names)
#     pick_result = plan_ik_seed_config(pick_pose, place_world, pick_approach_state, 0.15)
#     if pick_result is None:
#         raise RuntimeError("No solution found!")
#     pick_q = pick_result[0].cpu().numpy()
#     trajs_retract_pick = trajs_retract_pick_approach_result[0].clone().cpu().numpy()
#     trajs_retract_pick = np.vstack([trajs_retract_pick, pick_q])
#
#     ## 4.2 retract -> place_approach
#     ### Compute attach objects and states
#
#     place_result = plan_ik(place_pose, place_world, 0.15)
#     if place_result is None:
#         raise RuntimeError("No solution found!")
#     place_q = place_result[0].cpu().numpy().reshape(1, -1)
#     attached_state = JointState.from_position(place_result[0])
#     # attached_object = tube_mesh
#     start_state = JointState.from_position(robot_kin_model.retract_config.view(1, -1),
#                                            joint_names=robot_kin_model.joint_names)
#     attached_object_for_rendering = trimesh.Trimesh(attached_object.vertices.copy(), attached_object.faces.copy())
#     T = poses_to_frames(place_pose).reshape(4, 4)
#     invT = np.linalg.inv(T)
#     attached_object_for_rendering.apply_transform(invT)
#
#     ### Plan motion
#     trajs_retract_place_approach_result = plan_motion_attached_object(start_state, place_approach_pose, attached_object,
#                                                                       attached_state, place_world, 0.02)
#     if trajs_retract_place_approach_result is None:
#         raise RuntimeError("No solution found!")
#     trajs_retract_place = trajs_retract_place_approach_result[0].cpu().numpy()
#     trajs_retract_place = np.vstack([trajs_retract_place, place_q])
#
#     ### Resolve IK Jumping
#     # place_approach_state = JointState.from_position(trajs_retract_place_approach_result[0][-1],
#     #                                                 joint_names=robot_kin_model.joint_names)
#     # place_result = plan_ik_seed_config(place_pose, place_world, place_approach_state, 0.15)
#     # if place_result is None:
#     #     raise RuntimeError("No solution found!")
#     # place_q = place_result[0].cpu().numpy()
#     # trajs_retract_place = trajs_retract_place_approach_result[0].clone().cpu().numpy()
#     # trajs_retract_place = np.vstack([trajs_retract_place, place_q])
#
#     ## 4.3 pick_approach -> place_approach
#     pick_approach_q = torch.tensor(trajs_retract_pick[-2, :].reshape(1, -1), **tensor_args.as_torch_dict())
#     place_approach_q = torch.tensor(trajs_retract_place[-2, :].reshape(1, -1), **tensor_args.as_torch_dict())
#     start_state = JointState.from_position(pick_approach_q, joint_names=robot_kin_model.joint_names)
#     goal_state = JointState.from_position(place_approach_q, joint_names=robot_kin_model.joint_names)
#
#     trajs_pick_place_approach_result = plan_motion_js_attached_object(start_state, goal_state, attached_object,
#                                                                       attached_state, place_world, 0.15)
#     if trajs_pick_place_approach_result is None:
#         raise RuntimeError("No solution found!")
#     trajs_pick_place = trajs_pick_place_approach_result.cpu().numpy()
#     trajs_pick_place = np.vstack([pick_q, trajs_pick_place, place_q, trajs_pick_place[::-1], pick_q])
#
#     return trajs_pick_place, attached_object_for_rendering


def compute_pick_place(tube_id, tube_pose, tube_mesh, place_world, robot_kin_model):
    ## 2.1 Compute place poses
    place_result = plan_ik(tube_pose, place_world, 0.15)
    if place_result is None:
        raise RuntimeError("No solution found!")

    # update
    place_poses = place_result[1]

    ## 2.3 Plan Place IK
    ### Plan place approach IK
    place_approach_poses = place_poses.clone()
    place_approach_poses.position += torch.tensor([0, 0, 0.1], **tensor_args.as_torch_dict())
    place_approach_result = plan_ik(place_approach_poses, place_world, 0.15)
    if place_approach_result is None:
        raise RuntimeError("No solution found!")

    # update
    place_approach_poses = place_approach_result[1]
    place_poses = get_poses_by_flag(place_poses, place_approach_result[2])
    # print("Number of place poses", place_poses.position.shape[0])

    ### Compute place to pick transformation matrix
    pick_pose_pos = [-1.5, -1.5, 0]
    pick_pose_quat = [0, 1, -1, 0]
    pick_poses = Pose(position=torch.tensor([pick_pose_pos] * len(place_poses), **tensor_args.as_torch_dict()),
                      quaternion=torch.tensor([pick_pose_quat] * len(place_poses), **tensor_args.as_torch_dict()))
    pick_frames = poses_to_frames(pick_poses)

    ### Compute pick poses
    from curobo.geom.transform import matrix_to_quaternion
    pick_frames = torch.tensor(pick_frames, **tensor_args.as_torch_dict())
    pick_poses = Pose(position=pick_frames[:, :3, 3], quaternion=matrix_to_quaternion(pick_frames[:, :3, :3]))

    ### Plan pick ik
    pick_result = plan_ik(pick_poses, place_world, 0.15)
    if pick_result is None:
        raise RuntimeError("No solution found!")

    # update
    pick_poses = pick_result[1]
    place_poses = get_poses_by_flag(place_poses, pick_result[2])
    place_approach_poses = get_poses_by_flag(place_approach_poses, pick_result[2])

    ### Plan pick approach ik
    pick_approach_poses = pick_poses.clone()
    pick_approach_poses.position += torch.tensor([0, 0, 0.1], **tensor_args.as_torch_dict())
    pick_approach_result = plan_ik(pick_approach_poses, place_world, 0.15)
    if pick_approach_result is None:
        raise RuntimeError("No solution found!")

    # update
    pick_approach_poses = pick_approach_result[1]
    pick_poses = get_poses_by_flag(pick_poses, pick_approach_result[2])
    place_poses = get_poses_by_flag(place_poses, pick_approach_result[2])
    # print("Number of pick poses", place_poses.position.shape[0])

    ## 4.1 Retract -> pick_approach
    ### Choose one pick-place-approach pair

    pick_place_id = 0
    pick_pose = pick_poses.clone()[pick_place_id]
    place_pose = place_poses.clone()[pick_place_id]
    pick_approach_pose = pick_approach_poses.clone()[pick_place_id]
    place_approach_pose = place_approach_poses.clone()[pick_place_id]

    ### Compute attach objects and states

    mesh = trimesh.creation.cylinder(radius=0.024, height=0.75, sections=16)
    mesh.apply_translation(pick_pose_pos)
    mesh = Mesh(
        name=f"tube_mesh{tube_id}",
        pose=[*pick_pose_pos, *pick_pose_quat],
        vertices=mesh.vertices,
        faces=mesh.faces,
        color=[0.7, 0.7, 0.7, 1.0],
    )

    # pick_result = plan_ik(pick_pose, pick_world[3], 0.15)
    pick_result = plan_ik(pick_pose, place_world, 0.15)
    if pick_result is None:
        raise RuntimeError("No solution found!")
    pick_q = pick_result[0].cpu().numpy().reshape(1, -1)

    attached_state = JointState.from_position(pick_result[0])
    # attached_object = mesh
    attached_object = tube_mesh
    start_state = JointState.from_position(robot_kin_model.retract_config.view(1, -1),
                                           joint_names=robot_kin_model.joint_names)

    attached_object_for_rendering = trimesh.Trimesh(attached_object.vertices.copy(), attached_object.faces.copy())
    T = poses_to_frames(pick_pose).reshape(4, 4)
    invT = np.linalg.inv(T)
    attached_object_for_rendering.apply_transform(invT)

    ### Plan motion
    trajs_retract_pick_approach_result = plan_motion_attached_object(start_state, pick_approach_pose, attached_object,
                                                                     attached_state, place_world, 0.15)
    if trajs_retract_pick_approach_result is None:
        raise RuntimeError("No solution found!")
    # trajs_retract_pick = trajs_retract_pick_approach_result[0].clone().cpu().numpy()
    # trajs_retract_pick = np.vstack([trajs_retract_pick, pick_q])

    ### Resolve IK Jumping
    pick_approach_state = JointState.from_position(trajs_retract_pick_approach_result[0][-1],
                                                   joint_names=robot_kin_model.joint_names)
    pick_result = plan_ik_seed_config(pick_pose, place_world, pick_approach_state, 0.15)
    if pick_result is None:
        raise RuntimeError("No solution found!")
    pick_q = pick_result[0].cpu().numpy()
    trajs_retract_pick = trajs_retract_pick_approach_result[0].clone().cpu().numpy()
    trajs_retract_pick = np.vstack([trajs_retract_pick, pick_q])

    ## 4.2 retract -> place_approach
    ### Compute attach objects and states

    place_result = plan_ik(place_pose, place_world, 0.15)
    if place_result is None:
        raise RuntimeError("No solution found!")
    # place_q = place_result[0].cpu().numpy().reshape(1, -1)
    attached_state = JointState.from_position(place_result[0])

    start_state = JointState.from_position(robot_kin_model.retract_config.view(1, -1),
                                           joint_names=robot_kin_model.joint_names)

    attached_object_for_rendering = trimesh.Trimesh(attached_object.vertices.copy(), attached_object.faces.copy())
    T = poses_to_frames(place_pose).reshape(4, 4)
    invT = np.linalg.inv(T)
    attached_object_for_rendering.apply_transform(invT)

    ### Plan motion
    trajs_retract_place_approach_result = plan_motion_attached_object(start_state, place_approach_pose, attached_object,
                                                                      attached_state, place_world, 0.02)
    if trajs_retract_place_approach_result is None:
        raise RuntimeError("No solution found!")
    # trajs_retract_place = trajs_retract_place_approach_result[0].cpu().numpy()
    # trajs_retract_place = np.vstack([trajs_retract_place, place_q])

    ### Resolve IK Jumping
    place_approach_state = JointState.from_position(trajs_retract_place_approach_result[0][-1],
                                                    joint_names=robot_kin_model.joint_names)
    place_result = plan_ik_seed_config(place_pose, place_world, place_approach_state, 0.15)
    if place_result is None:
        raise RuntimeError("No solution found!")
    place_q = place_result[0].cpu().numpy()
    trajs_retract_place = trajs_retract_place_approach_result[0].clone().cpu().numpy()
    trajs_retract_place = np.vstack([trajs_retract_place, place_q])

    ## 4.3 pick_approach -> place_approach
    pick_approach_q = torch.tensor(trajs_retract_pick[-2, :].reshape(1, -1), **tensor_args.as_torch_dict())
    place_approach_q = torch.tensor(trajs_retract_place[-2, :].reshape(1, -1), **tensor_args.as_torch_dict())
    start_state = JointState.from_position(pick_approach_q, joint_names=robot_kin_model.joint_names)
    goal_state = JointState.from_position(place_approach_q, joint_names=robot_kin_model.joint_names)

    trajs_pick_place_approach_result = plan_motion_js_attached_object(start_state, goal_state, attached_object,
                                                                      attached_state, place_world, 0.15)
    if trajs_pick_place_approach_result is None:
        raise RuntimeError("No solution found!")
    trajs_pick_place = trajs_pick_place_approach_result.cpu().numpy()
    trajs_pick_place = np.vstack([pick_q, trajs_pick_place, place_q, trajs_pick_place[::-1], pick_q])

    return trajs_pick_place, attached_object_for_rendering