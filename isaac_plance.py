# show_planner.py

import time
from typing import Tuple, List

# 你的规划器类
from gridPathPlanner import GridPathPlanner

# Isaac Sim standalone app
from isaacsim import SimulationApp


# -------------------------------------------------
# 每个 roll 在「cell 局部坐标」中的起点/终点 (单位: cell 的长度, 假设 cell_size = 1m)
# cell 局部原点是该 cell 的左下角:
#   world_origin(cell_x, cell_y) = [cell_y * cell_size, -cell_x * cell_size, 0.0]
#   （注意：你已經改過軸，這裡我照著你這版保持不動）
# -------------------------------------------------
ROLL_LOCAL_POINTS = {
    0:  ([0.95, -0.05, 0.05], [0.95, -0.05, 0.95]),
    1:  ([0.95, -0.95, 0.05], [0.95, -0.95, 0.95]),
    2:  ([0.95, -0.05, 0.95], [0.95, -0.95, 0.05]),
    3:  ([0.95, -0.05, 0.05], [0.95, -0.95, 0.95]),
    4:  ([0.95, -0.05, 0.95], [0.95, -0.95, 0.95]),
    5:  ([0.95, -0.05, 0.95], [0.05, -0.05, 0.95]),
    6:  ([0.95, -0.95, 0.95], [0.05, -0.95, 0.95]),
    7:  ([0.05, -0.05, 0.05], [0.05, -0.05, 0.95]),
    8:  ([0.05, -0.95, 0.05], [0.05, -0.95, 0.95]),
    9:  ([0.05, -0.05, 0.95], [0.05, -0.95, 0.05]),
    10: ([0.05, -0.05, 0.05], [0.05, -0.95, 0.95]),
    11: ([0.05, -0.05, 0.95], [0.05, -0.95, 0.95]),
}


def cell_origin_world(cell_x: int, cell_y: int, cell_size: float) -> Tuple[float, float, float]:
    """
    返回某个 cell 左下角在世界坐标中的位置。
    world = cell_origin + local * cell_size

    注意：你自己的定義是 (cell_y, -cell_x)，我這裡保持一致：
      (cell_x, cell_y) = (0,0) -> [0,  0, 0]
      (cell_x, cell_y) = (0,1) -> [1,  0, 0]  (如果 cell_size = 1)
      (cell_x, cell_y) = (1,0) -> [0, -1, 0]
    """
    return cell_y * cell_size, -cell_x * cell_size, 0.0


def roll_world_points(
    cell_x: int,
    cell_y: int,
    floor_index: int,
    roll_index: int,
    cell_size: float,
    floor_height: float = None,
) -> Tuple[List[float], List[float]]:
    """
    把某个 cell、某一樓層中的 roll 的局部起点/终点，转成世界坐标。

    world = cell_origin + local * cell_size
    再加上樓層的 z 偏移：
      z += floor_index * floor_height
    """
    if roll_index not in ROLL_LOCAL_POINTS:
        raise ValueError(f"Invalid roll index: {roll_index}")

    if floor_height is None:
        floor_height = cell_size  # 預設一層高度 = cell_size

    local_start, local_end = ROLL_LOCAL_POINTS[roll_index]
    ox, oy, oz = cell_origin_world(cell_x, cell_y, cell_size)
    z_offset = floor_index * floor_height

    # DEBUG：你原來有 print origin，我也保留
    print("Origin (cell_x={}, cell_y={}, floor={}): {}".format(
        cell_x, cell_y, floor_index, (ox, oy, oz + z_offset))
    )

    ws = [
        ox + local_start[0] * cell_size,
        oy + local_start[1] * cell_size,
        oz + local_start[2] * cell_size + z_offset,
    ]
    we = [
        ox + local_end[0] * cell_size,
        oy + local_end[1] * cell_size,
        oz + local_end[2] * cell_size + z_offset,
    ]
    return ws, we


def main():
    # ------------ 规划参数 ------------
    grid_size = 5       # 5x5 网格
    cell_size = 1.0      # 每个 cell 尺寸 (m)
    step_time = 0.25     # 每根 roll 的时间间隔 (秒)
    num_floors = 2       # 樓層數，例如 2 層：先蓋第0層，再蓋第1層，再換下一個 cell

    # 这里放你要测试的 BLOCKED_CELLS（需与 GridPathPlanner 规则相容: 每一列至少一个 block）
    blocked_cells = [
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 3),
        (4, 2),
    ]
    # blocked_cells = [
    #     (0, 2),
    #     (1, 2),
    #     (2, 1),
    #     (3, 2),
    #     (4, 2),
    #     (5, 1),
    #     (6, 2),
    #     (7, 2),
    # ]

    # ------------ 先创建 Isaac Sim app ------------
    simulation_app = SimulationApp({"headless": False})

    # Debug draw extension
    from isaacsim.util.debug_draw import _debug_draw

    # 现在导入 debug_draw interface
    dd = _debug_draw.acquire_debug_draw_interface()

    # 尝试清除已有的线（如果 API 存在）
    for func_name in ("clear_lines", "clear_drawings", "clear"):
        func = getattr(dd, func_name, None)
        if callable(func):
            func()
            break

    # ------------ 用 GridPathPlanner 做计算 ------------
    planner = GridPathPlanner(size=grid_size, blocked_cells=blocked_cells)
    planner.print_info()

    sorted_path = planner.get_sorted_path()
    build_sequences = planner.get_build_sequences()

    # 展开成 “(cell_x, cell_y, floor_idx, roll_idx)” 的全局建造顺序
    # 規則：對每個 cell，按 floor 由低到高蓋完，再換下一個 cell
    build_plan: List[Tuple[int, int, int, int]] = []
    for (cell_x, cell_y), seq in zip(sorted_path, build_sequences):
        for floor_idx in range(num_floors):        # 0樓, 1樓, 2樓...
            for roll_idx in seq:                   # 該 cell 的 roll 序列
                build_plan.append((cell_x, cell_y, floor_idx, roll_idx))

    print("Sorted path (cells):", sorted_path)
    print(f"Total rolls to draw (all floors): {len(build_plan)}")

    # 线的颜色 (RGBA) 和宽度
    line_color_rgba = (0.0, 1.0, 0.0, 1.0)  # 绿色，不透明
    line_width = 2.0

    # ------------ 模拟循环，每隔 step_time 画一根 roll ------------
    next_roll_index = 0
    last_draw_time = time.time()
    all_done_time = None

    print("\nStart simulation loop, drawing one roll every {:.2f}s...".format(step_time))

    # 基本循环：每帧更新 Isaac Sim，然后用真实时间控制绘制节奏
    while simulation_app.is_running():
        simulation_app.update()

        now = time.time()

        # 还没画完所有 roll
        if next_roll_index < len(build_plan):
            if now - last_draw_time >= step_time:
                cell_x, cell_y, floor_idx, roll_idx = build_plan[next_roll_index]
                start_world, end_world = roll_world_points(
                    cell_x=cell_x,
                    cell_y=cell_y,
                    floor_index=floor_idx,
                    roll_index=roll_idx,
                    cell_size=cell_size,
                )

                # 转成 draw_lines 需要的列表形式
                start_list = [tuple(start_world)]         # List[Float3]
                end_list = [tuple(end_world)]             # List[Float3]
                color_list = [line_color_rgba]            # List[ColorRgba]
                size_list = [line_width]                  # List[float]

                dd.draw_lines(start_list, end_list, color_list, size_list)

                print(
                    f"cell ({cell_x}, {cell_y}), floor {floor_idx}, "
                    f"roll {roll_idx} from {start_world} to {end_world}"
                )

                next_roll_index += 1
                last_draw_time = now

                # 全部画完后，记一下时间，给你一点时间看
                if next_roll_index == len(build_plan):
                    all_done_time = now
        else:
            # 全部画完后，停留几秒再退出（這裡用 5 秒，比你之前 5000 秒實際一點）
            if all_done_time is not None and now - all_done_time > 5.0:
                print("All rolls drawn. Closing Isaac Sim.")
                break

    simulation_app.close()


if __name__ == "__main__":
    main()
