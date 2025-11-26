# torch
import torch
import numpy as np
import copy
import trimesh
# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml
from robofab.mqtt import publish_robot_trajs, publish_clear, publish_world, publish_frames, publish_grasp_object
from robofab.place import poses_to_frames
from curobo.geom.types import WorldConfig, Mesh, Cylinder, Cuboid
from curobo.geom.types import Pose
import pickle
from workplace.gen_scaffold_plan import get_design
from workplace.util import (load_kuka_kin_model,
                            tube_info,
                            gen_fully_placed_world,
                            tube_frame,
                            generate_rotated_frames
                            )
from ik import plan_ik, compute_pick_place

# convenience function to store tensor type and device
tensor_args = TensorDeviceType()

place_world = WorldConfig(
    cylinder=[
        Cylinder(
            name="col1",
            radius=0.4,
            height=1.2 * 2,
            pose=[3.2, -0.8, 1.2, np.sin(1.57 / 2), 0, 0, np.cos(1.57 / 2)],
            color=[0.8, 0.6, 0.4, 1.0],
        ),
        Cylinder(
            name="col2",
            radius=0.4,
            height=1.2 * 2,
            pose=[3.2, 0.8, 1.2, np.sin(1.57 / 2), 0, 0, np.cos(1.57 / 2)],
            color=[0.8, 0.6, 0.4, 1.0],
        ),
        # tube_cylinder[2]
    ],
    cuboid=[Cuboid(
        name="wall",
        dims=[2.0 * 2, 0.025 * 2, 2.4],
        pose=[3.2, 0, 1.2, np.sin(1.57 / 2), 0, 0, np.cos(1.57 / 2)],
        color=[0.8, 0.6, 0.4, 1.0],
    )]
)

if __name__ == "__main__":
    robot_config_file, robot_kin_model = load_kuka_kin_model(0.15)
    retract_q = robot_kin_model.retract_config.cpu().numpy().reshape(1, -1)
    # publish_clear()
    # publish_robot_trajs(retract_q, 0.15)

    xml, scaffold_structure, valid_tube_index = get_design()

    fully_placed_world, tube_mesh, tube_cylinder = gen_fully_placed_world(copy.deepcopy(scaffold_structure), valid_tube_index, place_world)
    # visualize
    publish_clear()
    publish_robot_trajs(retract_q, 0.15)
    publish_world(fully_placed_world, merged=True)

    tube_frames = tube_frame(scaffold_structure, valid_tube_index)
    init_frames = [i[np.newaxis, :, :] for i in tube_frames]
    init_frames = np.vstack(init_frames)
    publish_frames(init_frames)

    # tube_id = 16
    # rotated_frames, tube_poses = generate_rotated_frames(tube_frames[tube_id], num_steps=72)
    # publish_frames(rotated_frames)

    # place_result = plan_ik(tube_poses, place_world, 0.15)
    # traj, attached_object_for_rendering = compute_pick_place(tube_id, tube_poses[tube_id], tube_mesh[tube_id], place_world, robot_kin_model)
    # publish_robot_trajs(traj, 0.15)
    # publish_grasp_object(attached_object_for_rendering)

    # traj, attached_object_for_rendering = compute_pick_place(tube_id)

    # motion planning
    # final_trajs = np.empty((0, len(robot_kin_model.joint_names)))
    # save_data = {
    #     'seq': [],
    #     'traj': {},
    # }
    # for tube_id in range(len(tube_mesh)):
    #     try:
    #         rotated_frames, tube_poses = generate_rotated_frames(tube_frames[tube_id], num_steps=72)
    #         publish_frames(rotated_frames)
    #
    #         traj, attached_object_for_rendering = compute_pick_place(tube_id, tube_poses, tube_mesh[tube_id], place_world, robot_kin_model)
    #         final_trajs = np.vstack([final_trajs, traj])
    #         # update place world
    #         place_world = WorldConfig(
    #             cylinder=[*place_world.cylinder, tube_cylinder[tube_id]],
    #             cuboid=[*place_world.cuboid],
    #         )
    #         publish_robot_trajs(final_trajs, 0.15)
    #         publish_grasp_object(attached_object_for_rendering)
    #         publish_world(place_world, merged=True)
    #
    #         cur_traj = {
    #             'id': tube_id,
    #             'tube_traj': traj,
    #             'attached_object_for_rendering': attached_object_for_rendering
    #         }
    #
    #         save_data['traj'][tube_id] = cur_traj
    #         save_data['seq'].append(tube_id)
    #
    #         print(f"Successfully processed tube_id {tube_id}")
    #     except RuntimeError as e:
    #         print(f"Error processing tube_id {tube_id}: {e}")
    #
    # with open("save_data.pkl", "wb") as f:
    #     pickle.dump(save_data, f)


    with open("save_data.pkl", "rb") as f:
        save_data = pickle.load(f)


    visual_world = WorldConfig(
        cylinder=[
            Cylinder(
                name="col1",
                radius=0.4,
                height=1.2 * 2,
                pose=[3.2, -0.8, 1.2, np.sin(1.57 / 2), 0, 0, np.cos(1.57 / 2)],
                color=[0.8, 0.6, 0.4, 1.0],
            ),
            Cylinder(
                name="col2",
                radius=0.4,
                height=1.2 * 2,
                pose=[3.2, 0.8, 1.2, np.sin(1.57 / 2), 0, 0, np.cos(1.57 / 2)],
                color=[0.8, 0.6, 0.4, 1.0],
            ),
            # tube_cylinder[2]
        ],
        cuboid=[Cuboid(
            name="wall",
            dims=[2.0 * 2, 0.025 * 2, 2.4],
            pose=[3.2, 0, 1.2, np.sin(1.57 / 2), 0, 0, np.cos(1.57 / 2)],
            color=[0.8, 0.6, 0.4, 1.0],
        )]
    )

    for tube_id in save_data['seq']:
        traj = save_data['traj'][tube_id]['tube_traj']
        attached_object_for_rendering = save_data['traj'][tube_id]['attached_object_for_rendering']
        visual_world = WorldConfig(
        cylinder=[*visual_world.cylinder, tube_cylinder[tube_id]],
        cuboid=[*visual_world.cuboid],
        )
        publish_robot_trajs(traj, 0.15)
        publish_grasp_object(attached_object_for_rendering)
        publish_world(visual_world, merged=True)

