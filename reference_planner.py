import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

class ReferencePlanner:
    def __init__(self, grid_params):
        self.width = grid_params['width']
        self.height = grid_params['height']
        self.resolution = grid_params['resolution']
        self.origin_x = grid_params['origin_x']
        self.origin_y = grid_params['origin_y']
        print("ReferencePlanner initialized")

    def update_grid(self, obstacles, safety_margin):
        """ Creates a new grid and marks obstacle cells as unwalkable """
        self.matrix = np.ones((self.height, self.width), dtype=int)
        radius_in_cells = int(safety_margin / self.resolution)
        
        for obs_pos in obstacles:
            # Convert obstacle world position to grid position
            obs_grid_x, obs_grid_y = self.world_to_grid(obs_pos[0], obs_pos[1])
            
            # Mark a square of cells around the obstacle as unwalkable (0)
            for r in range(max(0, obs_grid_y - radius_in_cells), min(self.height, obs_grid_y + radius_in_cells + 1)):
                for c in range(max(0, obs_grid_x - radius_in_cells), min(self.width, obs_grid_x + radius_in_cells + 1)):
                    if (r - obs_grid_y)**2 + (c - obs_grid_x)**2 <= radius_in_cells**2:
                        if 0 <= r < self.height and 0 <= c < self.width:
                            self.matrix[r][c] = 0 # Mark as unwalkable
                            
        self.grid = Grid(matrix=self.matrix)

    def find_path(self, start_pos, goal_pos):
        """ Finds a path from start to goal on the current grid and returns it in world coordinates """
        start_gx, start_gy = self.world_to_grid(start_pos[0], start_pos[1])
        goal_gx, goal_gy = self.world_to_grid(goal_pos[0], goal_pos[1])

        goal_gx = np.clip(goal_gx, 0, self.width - 1)
        goal_gy = np.clip(goal_gy, 0, self.height - 1)

        # Ensure start and goal are within grid bounds and walkable
        if not (0 <= start_gx < self.width and 0 <= start_gy < self.height): return None
        if not (0 <= goal_gx < self.width and 0 <= goal_gy < self.height): return None
        
        start_node = self.grid.node(start_gx, start_gy)
        goal_node = self.grid.node(goal_gx, goal_gy)
        
        # Check if goal is walkable, if not, find nearest walkable node 
        if not self.grid.walkable(goal_gx, goal_gy):
            print("Warning: Goal is inside an obstacle. Pathfinding may fail")
            
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path_grid, runs = finder.find_path(start_node, goal_node, self.grid)

        if not path_grid:
            print("!!! Path not found! !!!")
            return None

        # Convert the path from grid coordinates back to world coordinates
        path_world = []
        for point in path_grid:
            wx, wy = self.grid_to_world(point.x, point.y)
            path_world.append([wx, wy, start_pos[2]]) # Keep Z constant
            
        print(f"Path found with {len(path_world)} points.")
        return np.array(path_world)

    def world_to_grid(self, world_x, world_y):
        """ Converts world coordinates to grid cell indices """
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """ Converts grid cell indices to world coordinates """
        world_x = (grid_x * self.resolution) + self.origin_x
        world_y = (grid_y * self.resolution) + self.origin_y
        return world_x, world_y