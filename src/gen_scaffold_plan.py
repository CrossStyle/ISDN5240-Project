import numpy as np
from typing import List, Dict
import xml.etree.ElementTree as ET
# 你的规划器类
from gridPathplanner_alg import GridPathPlanner

def get_geom_bounds(geom: Dict) -> Dict:
    """Compute world bounds for a geom based on type, pos, euler."""
    pos = geom['pos']
    size = geom['size']
    euler = geom['euler']

    # Rotation matrix from euler (xyz order)
    rx = np.array([[1, 0, 0], [0, np.cos(euler[0]), -np.sin(euler[0])], [0, np.sin(euler[0]), np.cos(euler[0])]])
    ry = np.array([[np.cos(euler[1]), 0, np.sin(euler[1])], [0, 1, 0], [-np.sin(euler[1]), 0, np.cos(euler[1])]])
    rz = np.array([[np.cos(euler[2]), -np.sin(euler[2]), 0], [np.sin(euler[2]), np.cos(euler[2]), 0], [0, 0, 1]])
    R = rz @ ry @ rx

    if geom['type'] == 'box':
        half_sizes = size
        corners_local = np.array(np.meshgrid(*[[-1, 1]] * 3, indexing='ij')).T.reshape(-1, 3) * half_sizes
        corners_world = (R @ corners_local.T).T + pos
        min_bounds = np.min(corners_world, axis=0)
        max_bounds = np.max(corners_world, axis=0)
    elif geom['type'] == 'cylinder':
        # Approximate bounds: radius in perpendicular planes, half-length along axis
        axis = R @ np.array([0, 0, 1])  # Axis direction
        perp1 = R @ np.array([1, 0, 0])
        perp2 = R @ np.array([0, 1, 0])
        half_length = size[1]
        radius = size[0]
        # Bounds along axis +/- half_length, perp +/- radius
        min_bounds = pos - half_length * np.abs(axis) - radius * (np.abs(perp1) + np.abs(perp2))
        max_bounds = pos + half_length * np.abs(axis) + radius * (np.abs(perp1) + np.abs(perp2))
    else:
        raise ValueError(f"Unsupported geom type: {geom['type']}")

    return {'min': min_bounds, 'max': max_bounds}


def compute_facade_contour(facade_geoms, depth=0.6, gap=0.2):
    """Compute segmented contour for scaffold: y segments with corresponding min_x for front."""
    y_bounds = set()
    for geom in facade_geoms:
        bounds = get_geom_bounds(geom)
        y_bounds.add(bounds['min'][1])
        y_bounds.add(bounds['max'][1])

    y_sorted = sorted(y_bounds)
    segments = []
    nx = []
    for i in range(len(y_sorted) - 1):
        y_min = y_sorted[i]
        y_max = y_sorted[i + 1]
        mid_y = (y_min + y_max) / 2
        active_min_x = np.inf
        for geom in facade_geoms:
            bounds = get_geom_bounds(geom)
            if bounds['min'][1] <= mid_y <= bounds['max'][1]:
                active_min_x = min(active_min_x, bounds['min'][0])
        if active_min_x < np.inf:
            segments.append({'y_min': y_min, 'y_max': y_max, 'min_x': active_min_x})
            nx.append(active_min_x)

    for seg in segments:
        min_x = seg['min_x']
        if min_x < max(nx):
            seg['min_x'] = max(nx) - depth - gap
        else:
            seg['min_x'] -= gap
    return segments


def plan_scaffold(facade_geoms: List[Dict], contour_segments: List[Dict], depth: float = 0.6, layer_height: float = 0.5, gap: float = 0.2) -> Dict:
    """Plan scaffold structure: sections, layers based on height."""
    # Max height from geoms
    max_height = max(get_geom_bounds(g)['max'][2] for g in facade_geoms)
    num_layers = int(np.ceil(max_height / layer_height))
    horizontal_levels = np.linspace(0, max_height, num_layers + 1)
    vertical_half_lengths = [layer_height / 2] * num_layers
    z_centers = (horizontal_levels[:-1] + horizontal_levels[1:]) / 2


    # ---- 重排 contour_segments ----
    # 依 min_x 由大到小排序；Python sort 是 stable，
    # 所以同一個 min_x 之間會保持原本的順序。

    contour_segments_sorted = sorted(
        contour_segments,
        key=lambda seg: seg["min_x"],
        reverse=True
    )


    sections = []
    for seg in contour_segments_sorted:
        sections.append({
            'y_start': seg['y_min'],
            'y_end': seg['y_max'],
            'x_front': seg['min_x'] - depth,
            'x_back': seg['min_x'],
            'add_diagonal': (seg['y_max'] - seg['y_min'] > 0.5), # only give true and false
        })

    return {
        'sections': sections,
        'num_layers': num_layers,                       # order of the layers
        'horizontal_levels': horizontal_levels,         # order of the layers
        'vertical_half_lengths': vertical_half_lengths, # order of the layers
        'z_centers': z_centers                          # order of the layers
    }


def generate_scaffold_xml(facade_xml_str: str,
                          gap: float = 0.2,
                          depth: float = 0.6,
                          layer_height: float = 1,
                          layer_z_low: list = [],
                          layer_z_high: list = [],
                          ):
    root = ET.fromstring(facade_xml_str)
    facade_geoms = []
    tube_info = {}
    for geom in root.findall('.//geom'):
        geom_dict = geom.attrib
        geom_dict['size'] = np.fromstring(geom_dict.get('size', '0 0 0'), sep=' ')
        geom_dict['pos'] = np.fromstring(geom_dict.get('pos', '0 0 0'), sep=' ')
        geom_dict['euler'] = np.fromstring(geom_dict.get('euler', '0 0 0'), sep=' ')
        facade_geoms.append(geom_dict)

    # --- 1) 原本的輪廓 & scaffold 計畫 ---
    contour_segments = compute_facade_contour(facade_geoms)
    plan = plan_scaffold(facade_geoms, contour_segments, depth, layer_height)

    # --- 2) 在這裡插入 GridPathPlanner：取得 sequence 並印出 ---
    # planner = GridPathPlanner(1.0, contour_segments, plan["num_layers"])
    # pos_sequences = planner.build_sequences_pos
    # 例如印第一個 segment 的所有 from/to
    # for i, (p_from, p_to) in enumerate(pos_sequences[0]):
    #     print(f"step {i}: from {p_from} to {p_to}")


    # planner.print_info()  # 這裡會印出 sorted segments 和每個 (layer, seg) 的 steps

    tube_radius = 0.024
    rgba = "0.7 0.7 0.7 1"
    material = "metal_mat"

    xml = '<?xml version="1.0" ?>\n<mujoco>\n'
    xml += '    <compiler angle = "radian"/>\n'
    xml += '    <option gravity="0 0 -9.81" timestep="0.002"/>\n'
    xml += '    <asset>\n'  # Assume assets are same
    xml += '        <texture name="wood" type="2d" builtin="flat" rgb1="0.8 0.6 0.4" rgb2="0.6 0.4 0.2" width="100" height="100"/>\n'
    xml += '        <material name="wood_mat" texture="wood" specular="0.5" shininess="0.5"/>\n'
    xml += '        <texture name="metal" type="2d" builtin="flat" rgb1="0.7 0.7 0.7" rgb2="0.5 0.5 0.5" width="100" height="100"/>\n'
    xml += '        <material name="metal_mat" texture="metal" specular="0.8" shininess="0.8"/>\n'
    xml += '        <texture name="groundplane" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>\n'
    xml += '        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n'
    xml += '    </asset>\n'
    xml += '    <worldbody>\n'
    xml += '        <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>\n'
    xml += '        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n'

    # Add facade geoms
    for geom in facade_geoms:
        pos_str = ' '.join(f"{p:.1f}" for p in geom['pos'])
        size_str = ' '.join(f"{s:.3f}" for s in geom['size'])
        euler_str = ' '.join(f"{e:.2f}" for e in geom['euler'])
        rgba_str = geom.get('rgba', '0.8 0.6 0.4 1')
        mat_str = geom.get('material', 'wood_mat')
        xml += f'        <geom name="{geom["name"]}" type="{geom["type"]}" size="{size_str}" pos="{pos_str}" euler="{euler_str}" rgba="{rgba_str}" material="{mat_str}"/>\n'

    tube_id = 0
    for sec in plan['sections']:
        y_start = sec['y_start']
        y_end = sec['y_end']
        x_front = sec['x_front']
        x_back = sec['x_back']
        add_diagonal = sec['add_diagonal']

        # Vertical poles
        for layer in range(plan['num_layers']):
            half_length = plan['vertical_half_lengths'][layer]
            z_center = plan['z_centers'][layer]
            # 0 
            from_p = [x_back, y_start, z_center - half_length +0.025]
            to_p = [x_back, y_start, z_center + half_length -0.025]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.3f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

            # 1
            from_p = [x_back, y_end, z_center - half_length+0.025]
            to_p = [x_back, y_end, z_center + half_length -0.025]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.3f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

            # 2 
            from_p = [x_back, y_end -0.05 , z_center + half_length -0.05 ]
            to_p = [x_back, y_start +0.05 ,z_center - half_length  +0.05 ]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.2f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

            # 3
            from_p = [x_back, y_end -0.025 , z_center + half_length ]
            to_p = [x_back, y_start +0.025 ,z_center + half_length ]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.2f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

            # 4
            from_p = [x_back-0.025, y_end , z_center + half_length ]
            to_p = [x_front+0.025, y_end ,z_center + half_length ]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.2f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

            # 5
            from_p = [x_back-0.025, y_start , z_center + half_length ]
            to_p = [x_front+0.025, y_start ,z_center + half_length ]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.2f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

            # 6
            from_p = [x_front, y_end , z_center - half_length +0.025]
            to_p = [x_front, y_end ,z_center + half_length  -0.025]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.2f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

            # 7
            from_p = [x_front, y_start , z_center - half_length +0.025]
            to_p = [x_front, y_start ,z_center + half_length  -0.025]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.2f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

            # 8
            from_p = [x_front, y_end -0.05 , z_center - half_length +0.05 ]
            to_p = [x_front, y_start +0.05 ,z_center + half_length  -0.05 ]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.2f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

            # 9
            from_p = [x_front, y_end -0.025 , z_center + half_length ]
            to_p = [x_front, y_start +0.025 ,z_center + half_length ]
            xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.3f} {from_p[1]:.3f} {from_p[2]:.3f} {to_p[0]:.3f} {to_p[1]:.3f} {to_p[2]:.2f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
            tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
            tube_id += 1

    # key: (x1,y1,z1,x2,y2,z2)（端點排序後、四捨五入），value: [tube_id1, tube_id2, ...]
    segment_dict = {}

    for tid, seg in tube_info.items():
        p1 = np.array(seg["from_p"], dtype=float)
        p2 = np.array(seg["to_p"], dtype=float)

        # 忽略方向：把端點按座標字典序排序，A→B 和 B→A 都變成同一個 key
        if tuple(p1) <= tuple(p2):
            a, b = p1, p2
        else:
            a, b = p2, p1

        # 四捨五入，避免 0.3000000004 之類的浮點數誤差
        key = tuple(np.round(np.concatenate([a, b]), 3))

        if key not in segment_dict:
            segment_dict[key] = []
        segment_dict[key].append(tid)

    overlapped_tube_ids = []
    valid_tube_ids_set = set()

    for key, ids in segment_dict.items():
        ids_sorted = sorted(ids)
        # 只要這個幾何 segment 出現超過一次，就視為重疊群組
        if len(ids_sorted) > 1:
            overlapped_tube_ids.extend(ids_sorted)
        # 無論有沒有重疊，這組裡「最小的 id」保留
        valid_tube_ids_set.add(ids_sorted[0])

    # 排序一下方便看
    overlapped_tube_ids = sorted(set(overlapped_tube_ids))
    valid_tube_index = sorted(valid_tube_ids_set)

    # 實際被丟掉的那幾根（重疊群組裡除了第一根以外）
    removed_tube_ids = sorted(set(overlapped_tube_ids) - valid_tube_ids_set)

    # print("Overlapped tube indices (all in overlapping groups):", overlapped_tube_ids)
    # print("Removed overlapped tube indices (discarded):", removed_tube_ids)
    print("Valid tube indices (kept):", valid_tube_index)


    xml += '    </worldbody>\n</mujoco>\n'
    return xml, tube_info


# get different designs by setting different wall, col1, and col2 locations
facade_xml = '''
<worldbody>
    <geom name="wall" type="box" size="2.5 0.025 3" pos="3.2 0 0" euler="0 0 1.57" rgba="0.8 0.6 0.4 1" material="wood_mat"/>
    <geom name="col1" type="cylinder" size="0.5 1.5" pos="3.2 -1 1.5" euler="0 0 1.57" rgba="0.8 0.6 0.4 1" material="wood_mat"/>
    <geom name="col2" type="cylinder" size="0.5 1.5" pos="3.2 1 1.5" euler="0 0 1.57" rgba="0.8 0.6 0.4 1" material="wood_mat"/>
</worldbody>
'''

layer_z_low = [0., 1., 2.]
layer_z_high = [1., 2., 3.]

# scaffold_structure[id] = {"start_pos":[], "end_pos": []}
xml, scaffold_structure = generate_scaffold_xml(
    facade_xml_str=facade_xml,
    layer_z_low=layer_z_low,
    layer_z_high=layer_z_high
)

visualize_using_mujoco = True
if visualize_using_mujoco:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model,data)as viewer:
        while viewer.is_running():
            viewer.sync()