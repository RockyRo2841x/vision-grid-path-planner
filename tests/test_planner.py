import heapq
import math
from pathlib import Path
import random
import tempfile
import unittest

import cv2
import numpy as np

from planner import a_star, extract_grid, run

IMAGE = Path(__file__).resolve().parents[1] / "examples/grid.jpeg"


def dijkstra(grid, start, goal):
    """Independent cost-only search used as the optimality reference."""
    queue, visited = [(0, start)], set()
    while queue:
        cost, point = heapq.heappop(queue)
        if point in visited:
            continue
        visited.add(point)
        if point == goal:
            return cost
        row, col = point
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            r, c = neighbor
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] is not None:
                heapq.heappush(queue, (cost + grid[r][c], neighbor))
    return math.inf


class PlannerTests(unittest.TestCase):
    def assert_valid_path(self, grid, start, goal, path, cost):
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertEqual(len(set(path)), len(path))
        self.assertEqual(sum(grid[r][c] for r, c in path[1:]), cost)
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            self.assertEqual(abs(r2 - r1) + abs(c2 - c1), 1)
            self.assertIsNotNone(grid[r2][c2])

    def test_supplied_image_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run(IMAGE, folder)
            self.assertEqual((result["rows"], result["columns"]), (6, 6))
            self.assertEqual(result["start"], (5, 5))
            self.assertEqual(result["destination"], (0, 0))
            self.assertEqual(result["total_cost"], 23)
            self.assertEqual(result["total_cost"], dijkstra(result["cost_grid"], (5, 5), (0, 0)))
            self.assert_valid_path(result["cost_grid"], (5, 5), (0, 0), result["path"], 23)
            self.assertIsNotNone(cv2.imread(str(Path(folder) / "path.png")))
            self.assertTrue((Path(folder) / "route.json").exists())

    def test_weighted_detour_beats_shorter_route(self):
        grid = [[0, 20, 0], [1, 1, 1]]
        path, cost = a_star(grid, (0, 0), (0, 2))
        self.assertEqual(cost, 3)
        self.assertEqual(len(path) - 1, 4)
        self.assert_valid_path(grid, (0, 0), (0, 2), path, cost)

    def test_random_grids_match_dijkstra(self):
        rng = random.Random(2841)
        for index in range(500):
            rows, cols = rng.randint(2, 7), rng.randint(2, 7)
            grid = [[rng.choice([None, 0, 0.5, 1, 2, 4, 8]) for _ in range(cols)] for _ in range(rows)]
            start, goal = (0, 0), (rows - 1, cols - 1)
            grid[0][0], grid[-1][-1] = 0, rng.choice([0, 0.5, 3])
            path, cost = a_star(grid, start, goal)
            with self.subTest(case=index):
                self.assertEqual(cost, dijkstra(grid, start, goal))
                if math.isfinite(cost):
                    self.assert_valid_path(grid, start, goal, path, cost)
                else:
                    self.assertEqual(path, [])

    def test_unreachable_and_blocked_endpoints(self):
        self.assertEqual(a_star([[0, None], [None, 0]], (0, 0), (1, 1)), ([], math.inf))
        self.assertEqual(a_star([[None, 0]], (0, 0), (0, 1)), ([], math.inf))

    def test_same_endpoint_and_zero_cost_cells(self):
        self.assertEqual(a_star([[7]], (0, 0), (0, 0)), ([(0, 0)], 0))
        path, cost = a_star([[0, 0, 0], [1, 1, 0]], (0, 0), (1, 2))
        self.assertEqual(cost, 0)
        self.assert_valid_path([[0, 0, 0], [1, 1, 0]], (0, 0), (1, 2), path, cost)

    def test_invalid_costs_shapes_and_coordinates(self):
        for grid in ([], [[]], [[0], [1, 2]], [[0, -1]], [[0, float("nan")]], [[0, math.inf]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                a_star(grid, (0, 0), (0, 0))
        for endpoint in (None, (-1, 0), (10, 10)):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                a_star([[0, 1]], (0, 0), endpoint)

    def test_missing_and_duplicate_markers(self):
        original = cv2.imread(str(IMAGE))
        missing = original.copy()
        missing[510:590, 510:590] = (255, 255, 255)
        with self.assertRaisesRegex(ValueError, "one start"):
            extract_grid(missing)
        duplicate = original.copy()
        duplicate[210:290, 410:490] = (1, 100, 0)
        with self.assertRaisesRegex(ValueError, "one start"):
            extract_grid(duplicate)

    def test_undetectable_grid_and_invalid_margin(self):
        with self.assertRaises(ValueError):
            extract_grid(np.full((100, 100, 3), 255, dtype=np.uint8))
        for margin in (-1, 100):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                extract_grid(cv2.imread(str(IMAGE)), margin)


if __name__ == "__main__":
    unittest.main()
