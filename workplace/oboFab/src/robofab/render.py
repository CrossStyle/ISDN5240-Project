import numpy as np
import polyscope as ps
import pytorch_kinematics as pk
import torch
from trimesh import Trimesh, Scene
from scipy.spatial.transform import Rotation as R

def draw_robot(name: str, robot_chain: pk.SerialChain, joint_angle: torch.tensor, **args):
    draw_part = args.get("draw_part", False)
    links = robot_chain.get_links()
    ret = robot_chain.forward_kinematics(joint_angle)
    scene = Scene()
    link_id = 0
    for link in links:
        for visual in link.visuals:
            if visual.geom_param is not None:
                mesh = visual.mesh
                new_mesh = Trimesh(mesh.vertices, mesh.faces)
                T = ret[link.name].get_matrix().numpy().reshape(4, 4)
                new_mesh.apply_transform(T)
                scene.add_geometry(new_mesh)
                if draw_part:
                    ps.register_surface_mesh(f"{link.name}", vertices=new_mesh.vertices, faces=new_mesh.faces)
                link_id = link_id + 1

    scene_mesh = scene.to_mesh()
    if not draw_part:
        ps.register_surface_mesh(f"{name}", vertices=scene_mesh.vertices, faces=scene_mesh.faces, color=(1, 1, 1, 1))
    return scene_mesh

def pose_to_matrix(pose):
    pos = pose[:3]
    quat = pose[3:7]
    rot = R.from_quat(quat, scalar_first = True).as_matrix()
    frame = np.eye(4)
    frame[0:3, 0:3] = rot
    frame[0:3, 3] = pos
    return frame

def draw_world_config(name:str, world_model, merge_meshes: bool = True):
    if world_model is not None:
        if merge_meshes:
            world_mesh = world_model.get_mesh_world(merge_meshes=True).mesh[0].get_trimesh_mesh()
            ps.register_surface_mesh(f"{name}_merged", vertices=world_mesh.vertices, faces=world_mesh.faces, color=(1, 1, 1, 1))
        else:
            for mesh in world_model.get_mesh_world(merge_meshes=False).mesh:
                T = pose_to_matrix(mesh.pose)
                trimesh_mesh = mesh.get_trimesh_mesh()
                trimesh_mesh.apply_transform(T)
                if mesh.color is not None:
                    ps.register_surface_mesh(f"{name}_{mesh.name}", vertices=trimesh_mesh.vertices, faces=trimesh_mesh.faces, color=mesh.color)
                else:
                    ps.register_surface_mesh(f"{name}_{mesh.name}", vertices=trimesh_mesh.vertices, faces=trimesh_mesh.faces)


def draw_frames(name: str, frames: list[np.ndarray]):
    o = []
    x, y, z = [], [], []
    for frame in frames:
        x.append(frame[:3, 0])
        y.append(frame[:3, 1])
        z.append(frame[:3, 2])
        o.append(frame[:3, 3])
    o = np.vstack(o)
    x = np.vstack(x)
    y = np.vstack(y)
    z = np.vstack(z)
    frame = ps.register_point_cloud(f"{name}_frame", o, color=(0, 0, 0, 1))
    frame.set_radius(0.02, relative=False)
    frame.add_vector_quantity("x", x, enabled=True, color=(1, 0, 0, 1), radius=0.02, length=0.1)
    frame.add_vector_quantity("y", y, enabled=True, color=(0, 1, 0, 1), radius=0.02, length=0.1)
    frame.add_vector_quantity("z", z, enabled=True, color=(0, 0, 1, 1), radius=0.02, length=0.1)

def draw_link_frames(name: str, robot_chain: pk.SerialChain, joint_angle: torch.tensor):
    links = robot_chain.get_links()
    ret = robot_chain.forward_kinematics(joint_angle)

    frames = []
    for link in links:
        for visual in link.visuals:
            if visual.geom_param is not None:
                T = ret[link.name].get_matrix().numpy().reshape(4, 4)
                frames.append(T)

    draw_frames(name, frames)

def draw_grasp_object(robot, q, grasp_object: Trimesh, tcp_name):
    T = robot.forward_kinematics(q)[tcp_name].get_matrix().numpy().reshape(4,4)
    object = Trimesh(vertices = grasp_object.vertices.copy(),
                     faces = grasp_object.faces.copy())
    object.apply_transform(T)
    ps.register_surface_mesh("grasp object", vertices=object.vertices, faces=object.faces, color=(1, 0.5, 0.5, 0.5))
    return

