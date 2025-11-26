from compas_eve import Publisher, Topic, Message
from compas_eve.mqtt import MqttTransport
import numpy as np
import time
import trimesh
from robofab.render import pose_to_matrix
import itertools

tx = MqttTransport(host="localhost", port=1885)

send_delay_time = 0.5

def publish_clear():
    topic = Topic("/robot/clear")
    publisher = Publisher(topic, transport=tx)
    msg = dict(text = "clear")
    publisher.publish(msg)
    time.sleep(send_delay_time)  # give some time for the message to be sent
    return

def publish_robot_trajs(trajs: np.array, finger_width = 0.02):
    trajs_finger = trajs.copy()
    if trajs.shape[1] == 6:
        trajs_finger = np.hstack([trajs, np.zeros((trajs.shape[0], 2)) + finger_width])  # add gripper open value
    topic = Topic(f"/robot/trajs")
    publisher = Publisher(topic, transport=tx)
    msg = dict(trajs = trajs_finger.tolist())
    publisher.publish(msg)
    time.sleep(send_delay_time)  # give some time for the message to be sent
    return

def publish_frames(frames: np.array):
    topic = Topic("/robot/frames")
    publisher = Publisher(topic, transport=tx)
    msg = dict(frames = frames.tolist())
    publisher.publish(msg)
    time.sleep(send_delay_time)
    return

def encoding_mesh(mesh):
    return {
        "vertices": mesh.vertices.copy(),
        "faces": mesh.faces.copy()
    }

def publish_grasp_object(mesh):
    topic = Topic("/robot/grasp_object")
    publisher = Publisher(topic, transport=tx)
    msg = dict(mesh=encoding_mesh(mesh))
    publisher.publish(msg)
    time.sleep(send_delay_time)

def publish_spheres(robot, q):
    spheres = robot.get_robot_as_spheres(q)
    spheres = list(itertools.chain(*spheres))
    sphere_coords = []
    for sph_data in spheres:
        sphere_coords.append([*sph_data.position, sph_data.radius])
    sphere_coords = np.array(sphere_coords)
    topic = Topic("/robot/spheres")
    publisher = Publisher(topic, transport=tx)
    publisher.publish(dict(spheres = sphere_coords))
    time.sleep(send_delay_time)
    return

def publish_world(world_model, merged = False):
    msg = dict(meshes = [], colors= [])
    if world_model is not None:
        if merged:
            world_mesh = world_model.get_mesh_world(merge_meshes=True).mesh[0].get_trimesh_mesh()
            msg["meshes"].append(encoding_mesh(world_mesh))
            msg["colors"].append([1, 1, 1, 1])
        else:
            for mesh in world_model.get_mesh_world(merge_meshes=False).mesh:
                T = pose_to_matrix(mesh.pose)
                trimesh_mesh = mesh.get_trimesh_mesh()
                trimesh_mesh.apply_transform(T)
                msg["meshes"].append(encoding_mesh(trimesh_mesh))
                if mesh.color is not None:
                    msg["colors"].append(mesh.color)
                else:
                    msg["colors"].append([1, 1, 1, 1])
    
    topic = Topic("/robot/world")
    publisher = Publisher(topic, transport=tx)
    publisher.publish(msg)
    time.sleep(send_delay_time * 4)
    return