import numpy as np
from typing import List, Dict
import xml.etree.ElementTree as ET


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

    sections = []
    for seg in contour_segments:
        sections.append({
            'y_start': seg['y_min'],
            'y_end': seg['y_max'],
            'x_front': seg['min_x'] - depth,
            'x_back': seg['min_x'],
            'add_diagonal': (seg['y_max'] - seg['y_min'] > 0.5),
        })

    return {
        'sections': sections,
        'num_layers': num_layers,
        'horizontal_levels': horizontal_levels,
        'vertical_half_lengths': vertical_half_lengths,
        'z_centers': z_centers
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

    contour_segments = compute_facade_contour(facade_geoms)
    plan = plan_scaffold(facade_geoms, contour_segments, depth, layer_height)

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
            for x, y in [(x_front, y_start), (x_front, y_end), (x_back, y_start), (x_back, y_end)]:
                from_p = [x, y, z_center - half_length]
                to_p = [x, y, z_center + half_length]
                xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.1f} {from_p[1]:.1f} {from_p[2]:.1f} {to_p[0]:.1f} {to_p[1]:.1f} {to_p[2]:.1f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
                tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
                tube_id += 1

        # x向横杆 (along x)
        for z in plan['horizontal_levels']:
            for y in [y_start, y_end]:
                from_p = [x_front, y, z]
                to_p = [x_back, y, z]
                xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.1f} {from_p[1]:.1f} {from_p[2]:.1f} {to_p[0]:.1f} {to_p[1]:.1f} {to_p[2]:.1f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
                tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
                tube_id += 1

        # y向横杆 (along y)
        for z in plan['horizontal_levels']:
            for x in [x_front, x_back]:
                from_p = [x, y_start, z]
                to_p = [x, y_end, z]
                xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.1f} {from_p[1]:.1f} {from_p[2]:.1f} {to_p[0]:.1f} {to_p[1]:.1f} {to_p[2]:.1f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
                tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
                tube_id += 1

        # 斜杆
        if add_diagonal:
            for layer in range(plan['num_layers']):
                z_low = layer_z_low[layer]
                z_high = layer_z_high[layer]
                for x in [x_front, x_back]:
                    from_p = [x, y_start, z_low]
                    to_p = [x, y_end, z_high]
                    xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.1f} {from_p[1]:.1f} {from_p[2]:.1f} {to_p[0]:.1f} {to_p[1]:.1f} {to_p[2]:.1f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
                    tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
                    tube_id += 1
                    from_p = [x, y_end, z_low]
                    to_p = [x, y_start, z_high]
                    xml += f'        <geom name="tube{tube_id}" type="cylinder" fromto="{from_p[0]:.1f} {from_p[1]:.1f} {from_p[2]:.1f} {to_p[0]:.1f} {to_p[1]:.1f} {to_p[2]:.1f}" size="{tube_radius}" rgba="{rgba}" material="{material}"/>\n'
                    tube_info[tube_id] = {"from_p": from_p, "to_p": to_p}
                    tube_id += 1

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
xml, scaffold_structure = generate_scaffold_xml(facade_xml_str=facade_xml, layer_z_low=layer_z_low, layer_z_high=layer_z_high)

visualize_using_mujoco = True
if visualize_using_mujoco:
    import mujoco
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    viewer.sync()
