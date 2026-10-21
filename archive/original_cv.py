import cv2
import numpy as np
import heapq

img_path = r"C:\Users\mvish\OneDrive\Desktop\NIT Trichy\Level UP\Python\CV\Grid_Image_InductionsTask1.jpeg"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError("Image is not found. ERROR!")

lower_grid = np.array([25, 25, 25])   
upper_grid = np.array([90, 90, 90])   

grid_mask = cv2.inRange(img, lower_grid, upper_grid)
cv2.imwrite("grid_mask.png", grid_mask)

def line_center(mask, direction):
    if direction == "horizontal":
        count = np.count_nonzero(mask, axis=1)
        threshold = mask.shape[1] * 0.8
    else:
        count = np.count_nonzero(mask, axis=0)
        threshold = mask.shape[0] * 0.8

    is_line = count > threshold

    centers = []
    on_line = False
    begin = 0
    
    for i in range(len(is_line)):
        
        if is_line[i] and not on_line:
            begin = i
            on_line = True

        elif not is_line[i] and on_line:
            end = i - 1
            on_line = False
            centers.append((begin + end) // 2)

    if on_line:
        end = len(is_line) - 1
        centers.append((begin + end) // 2)

    return centers

row_lines = line_center(grid_mask, "horizontal")
col_lines = line_center(grid_mask, "vertical")

rows = len(row_lines) - 1
cols = len(col_lines) - 1

print(f"Grid size: {rows} x {cols}")

reference_colours = {
    "obstacle": np.array([0, 0, 0]),
    "white": np.array([255, 255, 255]),
    "green": np.array([1, 255, 0]),
    "yellow": np.array([1, 255, 255]),
    "red": np.array([0, 0, 254]),
    "start": np.array([1, 100, 0]),
    "destination": np.array([254, 0, 255])
}

weight = {
    "white": 1, "green": 2, "yellow": 3, 
    "red": 4, "start": 0, "destination": 0, 
    "obstacle": None
}

def classify_color(avg_colour):
    best_name = None
    best_distance = float("inf")

    for name, ref_colour in reference_colours.items():
        distance = np.linalg.norm(avg_colour - ref_colour)
        if distance < best_distance:
            best_distance = distance
            best_name = name
    
    return best_name

tiles = []
tile_cost = []

start = None
destination = None

margin = 10

for r in range(rows):
    row_tile = []
    row_cost = []

    for c in range(cols):
        y1, y2 = row_lines[r] + margin, row_lines[r + 1] - margin
        x1, x2 = col_lines[c] + margin, col_lines[c + 1] - margin

        tile = img[y1:y2, x1:x2]
        avg_colour = np.array(cv2.mean(tile)[:3])
        tile_type = classify_color(avg_colour)

        row_tile.append(tile_type)
        row_cost.append(weight[tile_type])

        if tile_type == "start":
            start = (r, c)
        elif tile_type == "destination":
            destination = (r, c)

    tiles.append(row_tile)
    tile_cost.append(row_cost)

print("\nCost grid:")
for row in tile_cost:
    print(row)

print("\nStart:", start)
print("Destination:", destination)



#A* Pathfinding Algorithm
def manhattan_distance(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

def a_star(cost_grid, start, dest):
    if not start or not dest:
        return [], float("inf")

    rows, cols = len(cost_grid), len(cost_grid[0])
    distance = [[float("inf")] * cols for _ in range(rows)]
    previous = [[None] * cols for _ in range(rows)]
    distance[start[0]][start[1]] = 0

    pq = []
    
    heapq.heappush(pq, (manhattan_distance(start, dest), 0, start))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while pq:
        _, current_cost, current = heapq.heappop(pq)
        r, c = current

        if current == dest:
            break

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            
            if 0 <= nr < rows and 0 <= nc < cols and cost_grid[nr][nc] is not None:
                new_cost = current_cost + cost_grid[nr][nc]
                
                if new_cost < distance[nr][nc]:
                    distance[nr][nc] = new_cost
                    previous[nr][nc] = current
                    
                    f_score = new_cost + manhattan_distance((nr, nc), dest)
                    heapq.heappush(pq, (f_score, new_cost, (nr, nc)))

   
    path = []
    current = dest
    if distance[dest[0]][dest[1]] == float("inf"):
        return [], float("inf")

    while current is not None:
        path.append(current)
        current = previous[current[0]][current[1]]

    path.reverse()
    return path, distance[dest[0]][dest[1]]

path, total_cost = a_star(tile_cost, start, destination)

#path drawing
if not path:
    print("\nNo valid path found from Start to Destination!")
else:
    print(f"\nPath found with minimum cost: {total_cost}")
    print("Path coordinates:", path)

    output = img.copy()

    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        x1 = (col_lines[c1] + col_lines[c1 + 1]) // 2
        y1 = (row_lines[r1] + row_lines[r1 + 1]) // 2
        
        x2 = (col_lines[c2] + col_lines[c2 + 1]) // 2
        y2 = (row_lines[r2] + row_lines[r2 + 1]) // 2
        
        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 8)

    for r, c in path:
        center_x = (col_lines[c] + col_lines[c + 1]) // 2
        center_y = (row_lines[r] + row_lines[r + 1]) // 2
        
        cv2.circle(output, (center_x, center_y), 12, (255, 255, 255), -1)
        cv2.circle(output, (center_x, center_y), 12, (255, 0, 0), 3)

    cv2.imwrite("final_path_output.png", output)
    print("\nSuccess! Final image saved as final_path_output.png")