"""Extract a color-coded grid image and find a minimum-cost four-neighbor path."""

import argparse
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np

REFERENCE_COLOURS = {
    "obstacle": (0, 0, 0), "white": (255, 255, 255),
    "green": (1, 255, 0), "yellow": (1, 255, 255),
    "red": (0, 0, 254), "start": (1, 100, 0),
    "destination": (254, 0, 255),
}
WEIGHTS = {"white": 1, "green": 2, "yellow": 3, "red": 4,
           "start": 0, "destination": 0, "obstacle": None}


def line_center(mask, direction):
    if direction not in ("horizontal", "vertical"):
        raise ValueError("Line direction must be horizontal or vertical")
    axis = 1 if direction == "horizontal" else 0
    coverage = mask.shape[axis] * 0.8
    flags = np.count_nonzero(mask, axis=axis) > coverage
    centers, begin = [], None
    for index, flag in enumerate(flags):
        if flag and begin is None:
            begin = index
        elif not flag and begin is not None:
            centers.append((begin + index - 1) // 2)
            begin = None
    if begin is not None:
        centers.append((begin + len(flags) - 1) // 2)
    return centers


def classify_color(mean_bgr):
    return min(REFERENCE_COLOURS, key=lambda name:
               np.linalg.norm(np.asarray(mean_bgr) - REFERENCE_COLOURS[name]))


def extract_grid(image, margin=10):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a decoded three-channel BGR image")
    if margin < 0:
        raise ValueError("Cell margin must be nonnegative")
    mask = cv2.inRange(image, np.array([25, 25, 25], dtype=np.uint8),
                       np.array([90, 90, 90], dtype=np.uint8))
    row_lines = line_center(mask, "horizontal")
    col_lines = line_center(mask, "vertical")
    if len(row_lines) < 2 or len(col_lines) < 2:
        raise ValueError("Could not detect enough grid boundaries")
    tiles, starts, destinations = [], [], []
    for row in range(len(row_lines) - 1):
        cells = []
        for col in range(len(col_lines) - 1):
            y1, y2 = row_lines[row] + margin, row_lines[row + 1] - margin
            x1, x2 = col_lines[col] + margin, col_lines[col + 1] - margin
            if y1 >= y2 or x1 >= x2:
                raise ValueError("Cell margin leaves an empty sampling region")
            name = classify_color(cv2.mean(image[y1:y2, x1:x2])[:3])
            cells.append(name)
            if name == "start":
                starts.append((row, col))
            elif name == "destination":
                destinations.append((row, col))
        tiles.append(cells)
    if len(starts) != 1 or len(destinations) != 1:
        raise ValueError(f"Expected one start and one destination; found {len(starts)} and {len(destinations)}")
    return tiles, [[WEIGHTS[name] for name in row] for row in tiles], starts[0], destinations[0], row_lines, col_lines, mask


def manhattan_distance(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


def a_star(cost_grid, start, dest):
    """Minimize costs of entered cells; starting-cell cost is excluded.

    A minimum-cost simple path never needs to revisit the start for nonnegative
    costs. Excluding that move permits a lower bound using interior-cell costs
    and the destination entry cost, including a zero-cost destination.
    """
    if not cost_grid or not cost_grid[0] or any(len(row) != len(cost_grid[0]) for row in cost_grid):
        raise ValueError("Cost grid must be nonempty and rectangular")
    rows, cols = len(cost_grid), len(cost_grid[0])
    for row in cost_grid:
        for value in row:
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Traversable costs must be finite and nonnegative")
    for point in (start, dest):
        if point is None or len(point) != 2 or not all(isinstance(v, int) for v in point):
            raise ValueError("Start/destination must be integer (row, column) pairs")
        if not (0 <= point[0] < rows and 0 <= point[1] < cols):
            raise ValueError("Start/destination is outside the grid")
    start, dest = tuple(start), tuple(dest)
    if cost_grid[start[0]][start[1]] is None or cost_grid[dest[0]][dest[1]] is None:
        return [], math.inf
    if start == dest:
        return [start], 0
    interior = [value for row, values in enumerate(cost_grid) for col, value in enumerate(values)
                if value is not None and (row, col) not in (start, dest)]
    minimum = min(interior, default=0)
    goal_cost = cost_grid[dest[0]][dest[1]]

    def heuristic(point):
        if point == dest:
            return 0
        return max(0, manhattan_distance(point, dest) - 1) * minimum + goal_cost

    distances, previous = {start: 0}, {}
    queue = [(heuristic(start), 0, start)]
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != distances.get(current):
            continue
        if current == dest:
            path = [dest]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return path[::-1], cost
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row, col = current[0] + dr, current[1] + dc
            neighbor = (row, col)
            if not (0 <= row < rows and 0 <= col < cols) or neighbor == start:
                continue
            value = cost_grid[row][col]
            if value is None:
                continue
            candidate = cost + value
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor], previous[neighbor] = candidate, current
                heapq.heappush(queue, (candidate + heuristic(neighbor), candidate, neighbor))
    return [], math.inf


def draw_path(image, path, row_lines, col_lines):
    output = image.copy()
    points = [((col_lines[c] + col_lines[c + 1]) // 2,
               (row_lines[r] + row_lines[r + 1]) // 2) for r, c in path]
    for left, right in zip(points, points[1:]):
        cv2.line(output, left, right, (255, 0, 0), 8)
    for point in points:
        cv2.circle(output, point, 12, (255, 255, 255), -1)
        cv2.circle(output, point, 12, (255, 0, 0), 3)
    return output


def run(image_path, output_dir, margin=10):
    image_path, output_dir = Path(image_path), Path(output_dir)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    tiles, costs, start, goal, rows, cols, mask = extract_grid(image, margin)
    path, cost = a_star(costs, start, goal)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in (("grid-mask.png", mask), ("path.png", draw_path(image, path, rows, cols))):
        if (output_dir / filename).resolve() == image_path.resolve():
            raise ValueError("Output would overwrite the input image")
        if not cv2.imwrite(str(output_dir / filename), data):
            raise OSError(f"Could not save {filename}")
    result = {"image": image_path.name, "rows": len(costs), "columns": len(costs[0]),
              "coordinate_convention": "zero-based (row, column)",
              "movement": "four-neighbor", "cost_convention": "sum of entered-cell costs; start excluded",
              "tiles": tiles, "cost_grid": costs, "start": start, "destination": goal,
              "path": path, "moves": max(0, len(path) - 1),
              "reachable": bool(path), "total_cost": cost if path else None}
    (output_dir / "route.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=Path(__file__).parent / "examples/grid.jpeg")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--margin", type=int, default=10)
    args = parser.parse_args()
    try:
        result = run(args.image, args.output, args.margin)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    if result["reachable"]:
        print(f"Grid: {result['rows']}x{result['columns']}; moves: {result['moves']}; minimum cost: {result['total_cost']}")
    else:
        print("No route exists between the detected markers.")


if __name__ == "__main__":
    main()
