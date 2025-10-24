import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle as LegendCircle
import numpy as np
import time
from mpc_optimizer import MPCOptimizer, PerceptionModule  
from reference_planner import ReferencePlanner

class RealTimeVisualizer:
    def __init__(self, x_lim, y_lim, goal_position, all_world_obstacles, mpc_params, sensor_range):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.x_lim = x_lim
        self.y_lim = y_lim
        
        # Static
        goal_handle = self.ax.plot(goal_position[0], goal_position[1], 'g*', markersize=15)[0]
        
        safety_radius = mpc_params['safety_distance']
        obstacle_handles = []
        for obs in all_world_obstacles:
            circle = plt.Circle((obs[0], obs[1]), safety_radius, color='r', alpha=0.3)
            self.ax.add_artist(circle)
            obs_handle = self.ax.plot(obs[0], obs[1], 'rx', markersize=10)[0]
            obstacle_handles.append(obs_handle)

        # Dynamic
        self.executed_path_plot, = self.ax.plot([], [], 'b-o', markersize=4)
        self.mpc_plan_plot, = self.ax.plot([], [], 'm--')
        self.ref_path_plot, = self.ax.plot([], [], 'c:')
        self.sensor_circle = plt.Circle((0,0), sensor_range, color='gray', fill=False, ls=':')
        self.ax.add_artist(self.sensor_circle)

        safety_radius_proxy = LegendCircle((0,0), radius=0.1, color='r', alpha=0.3)
        sensor_range_proxy = Line2D([0], [0], linestyle=':', color='gray')
        
        # Define the exact handles and labels you want
        legend_handles = [goal_handle, obstacle_handles[0], self.executed_path_plot,
                          self.mpc_plan_plot, self.ref_path_plot, sensor_range_proxy, safety_radius_proxy]
        
        legend_labels = ['Goal', 'Obstacle', 'Executed Path', 'Current MPC Plan',
                          'Global Reference (A*)', 'Sensor Range', 'Safety Radius']
        
        self.ax.legend(handles=legend_handles, labels=legend_labels)

        self.ax.set_xlabel("X position (m)")
        self.ax.set_ylabel("Y position (m)")
        self.ax.set_title("Simulation of UAV Motion Planning using MPC")
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')

    def update(self, uav_pos, executed_path, mpc_plan, ref_path):
        """ Updates the plot with new data """
        executed_path = np.array(executed_path)
        self.executed_path_plot.set_data(executed_path[:, 0], executed_path[:, 1])
        self.mpc_plan_plot.set_data(mpc_plan[:, 0], mpc_plan[:, 1])
        self.ref_path_plot.set_data(ref_path[:, 0], ref_path[:, 1])
        
        self.sensor_circle.center = (uav_pos[0], uav_pos[1])
        
        self.fig.canvas.draw()
        plt.pause(0.01)

def run_simulation_matplotlib():
    print("--- Running MPC Simulation with Visualizer ---")

    # Parameters
    model_params = {'dt': 0.1, 'D_max': np.diag([0.5, 0.5, 0.5])}
    mpc_params = { 'P': 20, 'v_max': 2.0, 'a_max': 9.81, 'j_max': 1.0, 'safety_distance': 0.6 }
    cost_weights = { 'wt': 100.0, 'ws': 1.0, 'wc': 10000.0, 'wj': 0.5, 'alpha': 50.0, 'ref_speed': 1.0 }
    sensor_range = 1.5  # Sensor range in meters

    optimizer = MPCOptimizer(model_params, mpc_params, cost_weights)
    perception = PerceptionModule(sensor_range=sensor_range)

    grid_params = {
        'width': 100,      # cells
        'height': 50,      # cells
        'resolution': 0.1, # meters/cell
        'origin_x': -1.0,  # world coordinate of grid's bottom-left corner
        'origin_y': -2.5,
    }
    ref_planner = ReferencePlanner(grid_params)
    
    # Scenario Setup (Two Obstacles)
    current_state = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    goal_position = np.array([4.5, 2.5, 1.0])

    # For plot
    all_world_obstacles = [np.array([2.0, 2.0, 1.0]), np.array([4.0, 1.5, 1.0])]
    
    path_history = [current_state[0:3]]
    
    # Visualizer Setup
    x_lim = [-1, goal_position[0] + 1]
    y_lim = [-2, goal_position[1] + 2]
    visualizer = RealTimeVisualizer(x_lim=x_lim, y_lim=y_lim, 
                                    goal_position=goal_position, 
                                    all_world_obstacles=all_world_obstacles,
                                    mpc_params=mpc_params,
                                    sensor_range=sensor_range)

    # Main Simulation Loop
    num_sim_steps = 150
    for step in range(num_sim_steps):
        print(f"\n--- Simulation Step {step + 1}/{num_sim_steps} ---")

        # Check if goal is reached
        if np.linalg.norm(current_state[0:3] - goal_position) < 0.2:
            print("Goal reached!")
            break

        # Perceive
        uav_pos = current_state[0:3]
        visible_obstacles = perception.detect_obstacles(uav_pos, all_world_obstacles)
        print(f"Visible obstacles: {len(visible_obstacles)}")

        # GENERATE GLOBAL REFERENCE
        ref_planner.update_grid(visible_obstacles, mpc_params['safety_distance'] * 1.2) # Small buffer
        astar_path = ref_planner.find_path(uav_pos, goal_position)

        if astar_path is None:
            print("Pathfinder failed. Falling back to straight line reference")
            # If A* fails, just create a naive straight-line path to the goal
            direction_vector_loop = goal_position - current_state[0:3]
            direction_norm_loop = direction_vector_loop / np.linalg.norm(direction_vector_loop)
            global_ref_path = np.zeros((mpc_params['P'], 3))
            for i in range(mpc_params['P']):
                time_at_step = (i + 1) * model_params['dt']
                global_ref_path[i, :] = current_state[0:3] + direction_norm_loop * cost_weights['ref_speed'] * time_at_step
        else:
            # Check if the A* path reached the goal
            last_astar_point = astar_path[-1, :]
            dist_to_goal = np.linalg.norm(last_astar_point - goal_position)

            if dist_to_goal > 1.0: # If last point is more than 1m from goal
                print("A* path is local. Stitching straight line to goal")
                # Create a straight line segment from the end of A* to the goal
                num_stitch_points = int(dist_to_goal / 0.1) # Assuming path segments are ~0.1m
                if num_stitch_points < 2: num_stitch_points = 2

                stitch_path = np.linspace(last_astar_point, goal_position, num_stitch_points)
                global_ref_path = np.vstack([astar_path, stitch_path])
            else:
                global_ref_path = astar_path


        num_global_points = global_ref_path.shape[0]
        ref_path = np.zeros((mpc_params['P'], 3))
        for i in range(mpc_params['P']):
            # Find the closest point on the global path to where we expect to be
            lookahead_dist = cost_weights['ref_speed'] * i * model_params['dt']

            path_distances = np.linalg.norm(np.diff(global_ref_path, axis=0), axis=1)
            cumulative_dist = np.insert(np.cumsum(path_distances), 0, 0)
            
            # Find the index in the global path that corresponds to this lookahead distance
            target_index = np.searchsorted(cumulative_dist, lookahead_dist)
            if target_index >= num_global_points:
                target_index = num_global_points - 1
            
            ref_path[i, :] = global_ref_path[target_index, :]
        
        # PLAN
        optimized_trajectory = optimizer.solve(current_state, ref_path, visible_obstacles, cost_weights)
        if optimized_trajectory is None:
            print("Solver failed to find a solution. Halting")
            break
        
        # ACT
        current_state = optimized_trajectory[1, :]
        path_history.append(current_state[0:3])
        
        # VISUALIZE
        visualizer.update(uav_pos=current_state[0:3],
                          executed_path=path_history, 
                          mpc_plan=optimized_trajectory[:, 0:3], 
                          ref_path=global_ref_path)
        
        time.sleep(model_params['dt']) # Control the simulation speed

    print("\nSimulation finished. Close the plot window to exit.")
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    run_simulation_matplotlib()