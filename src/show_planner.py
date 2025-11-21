import time
import json
from typing import List, Tuple

from isaacsim import SimulationApp


def load_build_sequences_pos(json_path: str):
    """從 JSON 檔讀取 build_sequences_pos 結果。"""
    with open(json_path, "r") as f:
        data = json.load(f)

    build_sequences_pos = data["build_sequences_pos"]
    num_layers = data.get("num_layers", None)
    num_segments = data.get("num_segments", None)

    print(f"Loaded {len(build_sequences_pos)} sequences from {json_path}")
    print(f"  num_layers  : {num_layers}")
    print(f"  num_segments: {num_segments}")
    return build_sequences_pos


def flatten_plan(build_sequences_pos) -> List[Tuple[int, int, List[float], List[float]]]:
    """
    把每個 sequence 裡的 (from_xyz, to_xyz) 展平成一個列表：
    [(seq_idx, step_idx, from_xyz, to_xyz), ...]
    """
    build_plan = []
    for seq_idx, seq in enumerate(build_sequences_pos):
        for step_idx, pair in enumerate(seq):
            if not pair or len(pair) != 2:
                continue
            start_xyz, end_xyz = pair
            build_plan.append((seq_idx, step_idx, start_xyz, end_xyz))

    print(f"Total lines to draw: {len(build_plan)}")
    return build_plan


def main():
    # -------- 讀取 JSON 規劃結果 --------
    json_path = "build_sequences_pos.json"  # 跟你之前 save 的路徑一致
    build_sequences_pos = load_build_sequences_pos(json_path)
    build_plan = flatten_plan(build_sequences_pos)

    if len(build_plan) == 0:
        print("No lines to draw. Exit.")
        return

    # -------- 啟動 Isaac Sim --------
    simulation_app = SimulationApp({"headless": False})

    from isaacsim.util.debug_draw import _debug_draw

    dd = _debug_draw.acquire_debug_draw_interface()

    # 嘗試清空舊的繪製
    for func_name in ("clear_lines", "clear_drawings", "clear"):
        func = getattr(dd, func_name, None)
        if callable(func):
            func()
            break

    # 畫線設定
    step_time = 0.25  # 每條線間隔秒數
    line_color_rgba = (0.0, 1.0, 0.0, 1.0)  # 綠色
    line_width = 2.0

    next_index = 0
    last_draw_time = time.time()
    all_done_time = None

    print(f"\nStart drawing, one line every {step_time:.2f} seconds...\n")

    # -------- 模擬主迴圈 --------
    while simulation_app.is_running():
        simulation_app.update()
        now = time.time()

        if next_index < len(build_plan):
            if now - last_draw_time >= step_time:
                seq_idx, step_idx, start_xyz, end_xyz = build_plan[next_index]

                # Isaac debug_draw 需要 tuple
                start_list = [tuple(start_xyz)]
                end_list = [tuple(end_xyz)]
                color_list = [line_color_rgba]
                size_list = [line_width]

                dd.draw_lines(start_list, end_list, color_list, size_list)

                print(
                    f"[seq {seq_idx}, step {step_idx}] "
                    f"from {start_xyz} to {end_xyz}"
                )

                next_index += 1
                last_draw_time = now

                if next_index == len(build_plan):
                    all_done_time = now
        else:
            # 全部畫完後停留 5 秒關閉
            if all_done_time is not None and now - all_done_time > 5.0:
                print("All lines drawn. Closing Isaac Sim.")
                break

    simulation_app.close()


if __name__ == "__main__":
    main()
