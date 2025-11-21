# path.py
from collections import deque
from typing import List, Tuple


class GridPathPlanner:
    """
    规划在一个 N x N 网格中的“贴墙连续路径”并生成建造序列。
    (0, 0) 在左下角, (N-1, N-1) 在右上角。
    """

    def __init__(self, size: int, blocked_cells: List[Tuple[int, int]]):
        """
        :param size: 网格大小 N (表示 N x N)
        :param blocked_cells: 被阻塞的格子列表 [(x, y), ...]
        """
        self.size = size
        self.width = size
        self.height = size

        self._blocked_cells = list(blocked_cells)
        self._blocked_set = set(blocked_cells)

        # 在 init 里算好所有东西
        self._preferred_cells = self._compute_preferred_cells()
        self._full_path = self._build_full_path()
        self._sorted_path = self._sort_path_by_row_then_col()
        self._build_sequences = self._compute_build_sequences()

    # ========= 对外接口（get_...） =========

    def get_BLOCKED_CELLS(self) -> List[Tuple[int, int]]:
        return list(self._blocked_cells)

    def get_preferred(self) -> List[Tuple[int, int]]:
        return list(self._preferred_cells)

    def get_full_path(self) -> List[Tuple[int, int]]:
        return list(self._full_path)

    def get_sorted_path(self) -> List[Tuple[int, int]]:
        return list(self._sorted_path)

    def get_build_sequences(self) -> List[List[int]]:
        # 深拷贝一层，防止外部修改
        return [list(seq) for seq in self._build_sequences]

    def print_info(self) -> None:
        """
        打印所有信息和地图。
        """
        print(f"Grid size: {self.size} x {self.size}")
        print("Blocked cells:", self._blocked_cells)
        print("Preferred (x,y) per column:", self._preferred_cells)
        print("Path cells (in walking order):", self._full_path)
        print("Path cells sorted (y high→low, x left→right):", self._sorted_path)

        print("\nPer-cell build sequences (following sorted_path order):")
        for cell, seq in zip(self._sorted_path, self._build_sequences):
            print(f"  Cell {cell}: {seq}")

        print("\nGrid (top row is y=size-1, bottom is y=0):")
        self._draw_grid()

    # ========= 内部计算函数 =========

    def _compute_preferred_cells(self) -> List[Tuple[int, int]]:
        """
        对每一列 x，找到“最靠近 row=0 的墙下面一格”的目标点。
        如果下面没有 free，就往上找最近 free。
        返回列表 [(x0,y0), (x1,y1), ...]，长度等于网格宽度。
        """
        preferred = []

        for x in range(self.width):
            # 找出这一列所有 block 的 y
            ys = [y for (bx, y) in self._blocked_set if bx == x]
            if not ys:
                raise ValueError(f"Column {x} has no blocked cell, invalid map.")
            min_y = min(ys)

            # 尝试在 min_y 下面找 free（越靠近 min_y 越好）
            y_pref = None
            y = min_y - 1
            while y >= 0:
                if (x, y) not in self._blocked_set:
                    y_pref = y
                    break
                y -= 1

            # 如果下面没有 free，就往上找
            if y_pref is None:
                y = min_y + 1
                while y < self.height:
                    if (x, y) not in self._blocked_set:
                        y_pref = y
                        break
                    y += 1

            if y_pref is None:
                raise ValueError(f"Column {x} is fully blocked, no path possible.")

            preferred.append((x, y_pref))

        return preferred

    def _bfs_shortest_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        在 N x N 网格上，从 start 走到 goal 的最短路径（不走到 blocked）。
        只允许上下左右移动，返回包含 start 和 goal 的路径列表。
        """
        if start == goal:
            return [start]

        q = deque()
        q.append(start)
        visited = {start}
        parent = {start: None}

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while q:
            x, y = q.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if nxt in self._blocked_set:
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                parent[nxt] = (x, y)
                if nxt == goal:
                    # 回溯得到路径
                    path = [goal]
                    cur = goal
                    while parent[cur] is not None:
                        cur = parent[cur]
                        path.append(cur)
                    path.reverse()
                    return path
                q.append(nxt)

        raise ValueError(f"No path found from {start} to {goal}.")

    def _build_full_path(self) -> List[Tuple[int, int]]:
        """
        把每一列的“目标格子”串起来：
        从 col 0 的目标格子开始，依次用 BFS 连接到 col 1,2,... 的目标格子。
        """
        full_path: List[Tuple[int, int]] = []

        for i in range(len(self._preferred_cells) - 1):
            start = self._preferred_cells[i]
            goal = self._preferred_cells[i + 1]
            segment = self._bfs_shortest_path(start, goal)
            if i > 0:
                # 去掉重复的起点（上一段的终点）
                segment = segment[1:]
            full_path.extend(segment)

        return full_path

    def _sort_path_by_row_then_col(self) -> List[Tuple[int, int]]:
        """
        把路径排序：
        1) y 从大到小（row 高的在前）
        2) 在同一行里，x 从小到大（从左到右）

        例子：
        [(0,0), (1,0), (2,0), (3,0), (3,1), (4,1)]
        -> [(3,1), (4,1), (0,0), (1,0), (2,0), (3,0)]
        """
        return sorted(self._full_path, key=lambda c: (-c[1], c[0]))

    def _compute_build_sequences(self) -> List[List[int]]:
        """
        对 sorted_path 中的每一个 cell，给出它的「有效步骤序列」：
        - 默认一个 cell 有步骤 0..11
        - 如果左边 cell 已经 build：忽略 0,5,7  → 保留 1,2,3,4,6,8,9,10,11
        - 如果上面 cell 已经 build：忽略 0,1,2,3,4
        - 如果左边 & 上面都 build：忽略 0,1,2,3,4,5,7 → 保留 6,8,9,10,11

        返回形如：
        [
            [ ... first cell 的步骤 ... ],
            [ ... second cell 的步骤 ... ],
            ...
        ]
        顺序与 sorted_path 一致。
        """
        built = set()
        all_sequences: List[List[int]] = []

        for (x, y) in self._sorted_path:
            left_built = (x - 1, y) in built
            upper_built = (x, y + 1) in built

            ignore = set()
            if left_built and upper_built:
                # 同时有左边、上方
                ignore.update([0, 1, 2, 3, 4, 5, 7])
            elif left_built:
                # 只有左边
                ignore.update([0, 5, 7])
            elif upper_built:
                # 只有上方
                ignore.update([0, 1, 2, 3, 4])

            seq = [i for i in range(12) if i not in ignore]
            all_sequences.append(seq)

            # 当前 cell 标记为已 build
            built.add((x, y))

        return all_sequences

    def _draw_grid(self) -> None:
        """
        输出 N x N 地图：
        # = block
        * = free
        - = path
        path 覆盖在 free 上（不会覆盖 block）
        """
        path_set = set(self._full_path)

        grid = [['*' for _ in range(self.width)] for _ in range(self.height)]

        # 先标 block
        for (x, y) in self._blocked_set:
            grid[y][x] = '#'

        # 再标 path（不覆盖 block）
        for (x, y) in path_set:
            if grid[y][x] != '#':
                grid[y][x] = '-'

        # 从 y=height-1 打印到 y=0
        for y in range(self.height - 1, -1, -1):
            row_str = ''.join(grid[y])
            print(row_str)


# ========= 测试入口 =========

if __name__ == "__main__":
    # 示例：5x5 网格
    size = 5

    # 你当前给的例子（你可以随时改这里来测试）
    blocked_cells = [
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 3),
        (4, 2),
    ]

    planner = GridPathPlanner(size=size, blocked_cells=blocked_cells)

    # 打印所有信息和地图
    planner.print_info()

