import pytorch_kinematics as pk
import trimesh
from robofab import ROBOFAB_DATA_DIR
import torch
import numpy as np

def load_geometry(chain, prefix = "/robot/simple_robot"):
    for link in chain.get_links():
        for visual in link.visuals:
            mesh = None
            if visual.geom_type == "mesh":
                mesh_file = ROBOFAB_DATA_DIR + f"{prefix}/{visual.geom_param[0]}"
                mesh = trimesh.load_mesh(mesh_file)
            elif visual.geom_type == "box":
                mesh = trimesh.primitives.Box(extents=visual.geom_param)
            elif visual.geom_type == "cylinder":
                mesh = trimesh.primitives.Cylinder(radius=visual.geom_param[0], height=visual.geom_param[1])
            elif visual.geom_type == "sphere":
                mesh = trimesh.primitives.Sphere(radius=visual.geom_param)
            if mesh:
                V = torch.tensor(mesh.vertices, dtype=visual.offset.dtype)
                V = visual.offset.transform_points(V).cpu().numpy()
                visual.mesh = trimesh.Trimesh(V, mesh.faces)
    return chain

def load_franka_robot():
    urdf = ROBOFAB_DATA_DIR + f"/robot/franka/fr3_franka.urdf"
    chain = pk.build_chain_from_urdf(open(urdf, mode="rb").read())
    return load_geometry(chain, prefix = "/robot/franka")

def random_ee_frame(robot, ee_name = "tcp"):
    lb, ub = robot.get_joint_limits()
    lb = torch.tensor(lb, device=robot.device, dtype=robot.dtype)
    ub = torch.tensor(ub, device=robot.device, dtype=robot.dtype)
    q = torch.rand(robot.n_joints) * (ub - lb) + lb
    ee_frame = robot.forward_kinematics(q)[ee_name].get_matrix()
    return ee_frame.reshape(4, 4)