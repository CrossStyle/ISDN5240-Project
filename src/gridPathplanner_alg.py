import json
from typing import List, Dict
import math
import numpy as np

# 每個步驟編號 -> (local_from, local_to)
ROLL_LOCAL_POINTS = {
    0:  ([0.95, 0.05, 0.05], [0.95, 0.05, 0.95]),
    2:  ([0.95, 0.05, 0.95], [0.95, 0.95, 0.05]),
    3:  ([0.95, 0.05, 0.05], [0.95, 0.95, 0.95]),
    4:  ([0.95, 0.05, 0.95], [0.95, 0.95, 0.95]),
    5:  ([0.95, 0.05, 0.95], [0.05, 0.05, 0.95]),
    6:  ([0.95, 0.95, 0.95], [0.05, 0.95, 0.95]),
    7:  ([0.05, 0.05, 0.05], [0.05, 0.05, 0.95]),
    8:  ([0.05, 0.95, 0.05], [0.05, 0.95, 0.95]),
    9:  ([0.05, 0.05, 0.95], [0.05, 0.95, 0.05]),
    10: ([0.05, 0.05, 0.05], [0.05, 0.95, 0.95]),
    11: ([0.05, 0.05, 0.95], [0.05, 0.95, 0.95]),
}



class GridPathPlanner:
    """
    極簡版本：
    - 不再有網格、不再有 cell。
    - 只接收 segments + num_layers。
    - 可以先用 size_sub_segments 把每個 segment 沿 y 方向切成較小子段。
    """

    def __init__(self, size_sub_segments: float, segments: List[Dict], num_layers: int):
        """
        :param size_sub_segments: 子 segment 的 y 長度，例如 1.0 代表 1m
        :param segments: compute_facade_contour 得到的段列表，例如：
            [
                {"y_min": -2.5, "y_max": -1.3, "min_x": 2.6},
                {"y_min": -1.3, "y_max":  0.2, "min_x": 2.8},
                ...
            ]
        :param num_layers: 高度層數
        """
        if not segments:
            raise ValueError("segments 不能是空的")

        self.size_sub_segments: float = float(size_sub_segments)

        self.raw_segments: List[Dict] = segments
        # 先把原始 segments 按照 size_sub_segments 沿 y 方向切成小段
        self.pre_segments: List[Dict] = self._subdivide_segments(segments)

        # 高度層數
        self.num_layers: int = int(num_layers)

        # 排序後的 segments
        self.sorted_segments: List[Dict] = self._sort_segments_by_x_then_y()

        # 給每個 segment 多加 origin 資訊
        # x_origin = min_x - size_sub_segments
        # y_origin = y_min
        # z_origin = 0.0 (先假設在地面)
        for seg in self.sorted_segments:
            seg["x_origin"] = seg["min_x"] - self.size_sub_segments
            seg["y_origin"] = seg["y_min"]
            seg["z_origin"] = 0.0

        # 依 build sequence + origin 算出每一步的世界座標 from/to
        self.build_sequences_pos = self._compute_build_sequences_pos()

        self.save_build_sequences_pos("build_sequences_pos.json")




    # ---------------- 新增的子函數：把 segments 切成小段 ----------------
    def _subdivide_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        把每個 segment 沿 y 方向切成長度約為 size_sub_segments 的小段。
        只保留需要的欄位：y_min, y_max, min_x

        改良版：
        - 不再用「while cur+step < y_max」的方式製造零頭
        - 而是先決定要切成幾段 n_sub ≈ length/step，再把整段平均切成 n_sub 等份
        - 這樣就不會出現 y_min == y_max 或極小殘段
        """
        import math

        step = float(self.size_sub_segments)
        if step <= 0.0:
            raise ValueError("size_sub_segments 必須 > 0")

        new_segments: List[Dict] = []
        eps = 1e-6  # 判斷「太短」的容差

        for seg in segments:
            y_min = float(seg["y_min"])
            y_max = float(seg["y_max"])
            x_min = float(seg["min_x"])

            length = y_max - y_min
            if length <= eps:
                # 原始段本身幾乎沒有長度，直接略過
                continue

            # 目標是讓 segment 長度大約是 step：
            #   n_sub ≈ length / step
            # 用 round 避免像 1.00079 被切成 2 段
            n_sub = int(round(length / step))
            if n_sub < 1:
                n_sub = 1

            # 平均切成 n_sub 等份
            seg_len = length / n_sub

            for i in range(n_sub):
                local_y_min = y_min + i * seg_len
                local_y_max = y_min + (i + 1) * seg_len

                if local_y_max - local_y_min <= eps:
                    continue

                new_segments.append(
                    {
                        "y_min": local_y_min,
                        "y_max": local_y_max,
                        "min_x": x_min,
                    }
                )

        return new_segments


    # -------------------------------------------------------------
    # 以下邏輯保持不變
    # -------------------------------------------------------------
    def _sort_segments_by_x_then_y(self) -> List[Dict]:
        """
        排序規則：
        1. 先依照 x_min 由大到小
        2. 如果 x_min 一樣，再依照 y_min 由小到大
        """
        return sorted(
            self.pre_segments,
            key=lambda s: (-s["min_x"], s["y_min"])
        )

    def compute_build_sequences(self) -> List[List[int]]:
        """
        對每個 (layer, segment) 計算 build steps。

        左邊已建的定義：
          在同一個 layer 內，只要有任何已建過的 segment，
          同時滿足：
            1) prev_seg.y_max == cur_seg.y_min
            2) prev_seg.min_x == cur_seg.min_x
          就算「左邊有結構」。
        """
        sequences: List[List[int]] = []
        built = set()  # 已建好的單元：key = (layer, seg_idx)

        for layer in range(self.num_layers):
            for seg_idx, seg in enumerate(self.sorted_segments):
                cur_y_min = seg["y_min"]
                cur_x_min = seg["min_x"]

                # 檢查左邊：同一層 + 同一條 x 線 + y 有貼齊
                left_built = False
                for (b_layer, b_seg_idx) in built:
                    if b_layer != layer:
                        continue

                    prev_seg = self.sorted_segments[b_seg_idx]
                    if (
                        math.isclose(
                            prev_seg["y_max"], cur_y_min,
                            rel_tol=1e-6,
                            abs_tol=1e-6,
                        )
                        and math.isclose(
                            prev_seg["min_x"], cur_x_min,
                            rel_tol=1e-6,
                            abs_tol=1e-6,
                        )
                    ):
                        left_built = True
                        break

                ignore = set()
                if left_built:
                    # 左邊有結構就忽略這幾個 step（你之前的規則）
                    ignore.update([0, 5, 7])

                steps = [i for i in range(12) if i not in ignore]
                sequences.append(steps)

                built.add((layer, seg_idx))

        return sequences

    def print_info(self) -> None:
        """
        簡單印出：
        - sorted segments 資訊（包含 origin）
        - 每個 (layer, seg) 的 steps 序列
        """
        print("===== GridPathPlanner Info =====")
        print(f"num_layers = {self.num_layers}")
        print(f"num_segments = {len(self.sorted_segments)}\n")
        print(f"orignal segments list (before subdivide):")
        for i, seg in enumerate(self.raw_segments):
            print(
                f"seg {i}: "
                f"x_min = {seg['min_x']:.3f}, "
                f"y = [{seg['y_min']:.3f}, {seg['y_max']:.3f}]"
            )

        print("=== Sorted segments ===")
        for i, seg in enumerate(self.sorted_segments):
            print(
                f"seg {i}: "
                f"x_min = {seg['min_x']:.3f}, "
                f"y = [{seg['y_min']:.3f}, {seg['y_max']:.3f}], "
                f"origin = ({seg['x_origin']:.3f}, {seg['y_origin']:.3f}, {seg['z_origin']:.3f})"
            )

        print("\n=== Build sequences (by layer, seg) ===")
        sequences = self.compute_build_sequences()
        idx = 0
        for layer in range(self.num_layers):
            for seg_idx, seg in enumerate(self.sorted_segments):
                steps = sequences[idx]
                print(
                    f"layer={layer}, seg={seg_idx}, "
                    f"y=[{seg['y_min']:.3f}, {seg['y_max']:.3f}], "
                    f"steps={steps}"
                )
                idx += 1
        print("===== End GridPathPlanner Info =====\n")

    def _compute_build_sequences_pos(self):
        """
        使用 compute_build_sequences() 的結果 + 每個 segment 的 origin +
        ROLL_LOCAL_POINTS 將每一個步驟編號(0~11)轉成世界座標的 (from_xyz, to_xyz)。

        回傳：
            List[List[Tuple[from_xyz, to_xyz]]]
        - 外層 index：跟 compute_build_sequences() 回傳的序列一一對應
        - 內層 list：這個序列中每個 step 的世界座標 from/to
        """
        sequences = self.compute_build_sequences()
        num_segments = len(self.sorted_segments)
        all_pos = []

        if num_segments == 0:
            return all_pos

        # 假設每層 z 間距 = 1.0（對應你 scaffold 的 layer_z_low = [0,1,2,...]）
        z_step = 1.0

        for global_idx, step_ids in enumerate(sequences):
            # 推回這個 sequence 對應的是哪一層、哪一個 segment
            layer = global_idx // num_segments
            seg_idx = global_idx % num_segments
            seg = self.sorted_segments[seg_idx]

            # 這個 segment 在該層的「世界原點」
            origin = np.array(
                [
                    seg["x_origin"],
                    seg["y_origin"],
                    seg["z_origin"] + layer * z_step,
                ],
                dtype=float,
            )

            step_pos_list = []
            for step_id in step_ids:
                # 這個步驟對應的局部起點/終點
                if step_id not in ROLL_LOCAL_POINTS:
                    # 理論上都在 0~11，不在的就略過
                    continue
                local_from, local_to = ROLL_LOCAL_POINTS[step_id]
                local_from = np.array(local_from, dtype=float)
                local_to = np.array(local_to, dtype=float)

                world_from = (origin + local_from).tolist()
                world_to = (origin + local_to).tolist()
                step_pos_list.append((world_from, world_to))

            all_pos.append(step_pos_list)

        return all_pos
    
    def save_build_sequences_pos(self, filepath: str) -> None:
        """
        將 build_sequences_pos 存成 JSON 檔，方便其他 Python 腳本載入使用。

        JSON 結構大致為：
        {
            "num_layers": int,
            "num_segments": int,
            "build_sequences_pos": [
                [  # 第 0 個 sequence
                    [[from_x, from_y, from_z], [to_x, to_y, to_z]],
                    ...
                ],
                [  # 第 1 個 sequence
                    ...
                ],
                ...
            ]
        }
        """
        import json
        import os

        # 確保資料夾存在
        dir_name = os.path.dirname(filepath)
        if dir_name != "":
            os.makedirs(dir_name, exist_ok=True)

        data = {
            "num_layers": self.num_layers,
            "num_segments": len(self.sorted_segments),
            "build_sequences_pos": self.build_sequences_pos,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)