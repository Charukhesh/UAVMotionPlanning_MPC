import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle as LegendCircle

from mpc_optimizer import MPCOptimizer, PerceptionModule, NoisyPerceptionModule
from reference_planner import ReferencePlanner
from risk_aware_optimizer import RiskAwareMPCOptimizer
import time

class ComparisonVisualizer:
    """A class to handle real-time plotting of TWO drones simultaneously"""
    def __init__(self, x_lim, y_lim, goal_position, all_world_obstacles, noisy_safety, safety):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        
        goal_handle = self.ax.plot(goal_position[0], goal_position[1], 'g*', markersize=20)
        obstacle_handles = []
        for obs in all_world_obstacles:
            circle1 = plt.Circle((obs[0], obs[1]), noisy_safety, color='r', alpha=0.3)
            circle2 = plt.Circle((obs[0], obs[1]), safety, color='y', alpha=0.3)
            self.ax.add_artist(circle1)
            self.ax.add_artist(circle2)
            obs_handle = self.ax.plot(obs[0], obs[1], 'rx', markersize=15)
            obstacle_handles.append(obs_handle)

        self.baseline_path_plot, = self.ax.plot([], [], 'r-o', markersize=4)
        self.riskaware_path_plot, = self.ax.plot([], [], 'b-o', markersize=4)
        self.mpc_plan_plot, = self.ax.plot([], [], 'm--')

        safety_radius_noisy = LegendCircle((0,0), radius=0.1, color='r', alpha=0.3)
        safety_radius = LegendCircle((0,0), radius=0.1, color='y', alpha=0.3)

        legend_handles = [goal_handle, obstacle_handles[0], self.baseline_path_plot,
                          self.riskaware_path_plot, self.mpc_plan_plot, safety_radius_noisy, safety_radius]
        
        legend_labels = ['Goal', 'Obstacle', 'Baseline Path', 'RiskAware Path', 'RiskAware MPC Plan',
                        'Safety Radius Noisy', 'Risk Aware Safety Radius']
        
        self.ax.legend(handles=legend_handles, labels=legend_labels)

        self.ax.set_title("Comparison: Baseline vs. Risk Aware MPC", fontsize=16)
        self.ax.set_xlabel("X position (m)"); self.ax.set_ylabel("Y position (m)")
        self.ax.set_xlim(x_lim); self.ax.set_ylim(y_lim)
        self.ax.grid(True); self.ax.set_aspect('equal', adjustable='box')

    def update(self, baseline_history, riskaware_history, riskaware_plan):
        baseline_path = np.array(baseline_history)
        riskaware_path = np.array(riskaware_history)
        
        self.baseline_path_plot.set_data(baseline_path[:, 0], baseline_path[:, 1])
        self.riskaware_path_plot.set_data(riskaware_path[:, 0], riskaware_path[:, 1])
        self.mpc_plan_plot.set_data(riskaware_plan[:, 0], riskaware_plan[:, 1])
        
        self.fig.canvas.draw()
        plt.pause(0.01)

def race_simulation():
    print("--- Running Live Comparison: Baseline vs. Risk Aware MPC ---")

    model_params = {'dt': 0.1, 'D_max': np.diag([0.5, 0.5, 0.5])}
    cost_weights = { 'wt': 5.0, 'ws': 1.0, 'wc': 10000.0, 'wj': 1.0, 'alpha': 50.0, 'ref_speed': 1.0 }

    noise_std_dev = 0.15
    baseline_mpc_params = { 'P': 20, 'v_max': 2.0, 'a_max': 9.81, 'j_max': 1.0, 'safety_distance': 0.6 + 1.5 * noise_std_dev}
    risk_aware_mpc_params = {'P': 20, 'v_max': 2.0, 'a_max': 9.81, 'j_max': 1.0, 'safety_distance': 0.6, 'max_risk': 0.01 }
    
    baseline_optimizer = MPCOptimizer(model_params, baseline_mpc_params, cost_weights)
    riskaware_optimizer = RiskAwareMPCOptimizer(model_params, risk_aware_mpc_params, cost_weights)

    sensor_range = 2.0
    goal_position = np.array([10.0, 2.5, 1.0])
    all_world_obstacles = [
        np.array([2.0, 2.0, 1.0]), np.array([4.0, 0.3, 1.0]),
        np.array([6.0, 1.8, 1.0]), np.array([8.0, 1.5, 1.0]),
        np.array([5.0, 3.5, 1.0])
    ]
    grid_params = {'width': 120, 'height': 60, 'resolution': 0.1, 'origin_x': -1.0, 'origin_y': -2.0}
    
    noisy_perception = NoisyPerceptionModule(sensor_range=sensor_range, noise_std_dev=noise_std_dev)
    ref_planner = ReferencePlanner(grid_params)

    baseline_state = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    riskaware_state = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    baseline_history = [baseline_state[0:3]]
    riskaware_history = [riskaware_state[0:3]]

    baseline_time = float('inf')
    riskaware_time = float('inf')
    baseline_finished = False
    riskaware_finished = False

    visualizer = ComparisonVisualizer(
        x_lim=[-1, 11], y_lim=[-1, 5],
        goal_position=goal_position, all_world_obstacles=all_world_obstacles,
        noisy_safety=baseline_mpc_params['safety_distance'],
        safety=risk_aware_mpc_params['safety_distance']
    )
    
    # Main Simulation Loop
    num_sim_steps = 350
    uav_pos = baseline_state[0:3]
    for step in range(num_sim_steps):
        current_time = step * model_params['dt']
        print(f"\n--- Step {step+1}/{num_sim_steps} ---")

        if not baseline_finished:
            print("  - Planning for Baseline (Timid)...")
            base_uav_pos = baseline_state[0:3]
            
            visible_obstacles_noisy = noisy_perception.detect_obstacles(base_uav_pos, all_world_obstacles)
            visible_obstacle_means = [obs[0] for obs in visible_obstacles_noisy]

            ref_planner.update_grid(visible_obstacle_means, baseline_mpc_params['safety_distance'])
            astar_path = ref_planner.find_path(base_uav_pos, goal_position)
            
            if astar_path is None:
                direction_vector_loop = goal_position - uav_pos
                direction_norm_loop = direction_vector_loop / np.linalg.norm(direction_vector_loop)
                global_ref_path = np.zeros((baseline_mpc_params['P'], 3))
                for i in range(baseline_mpc_params['P']):
                    time_at_step = (i + 1) * model_params['dt']
                    global_ref_path[i, :] = uav_pos + direction_norm_loop * cost_weights['ref_speed'] * time_at_step
            else:
                last_astar_point = astar_path[-1, :]; dist_to_goal = np.linalg.norm(last_astar_point - goal_position)
                if dist_to_goal > 1.0:
                    num_stitch = int(dist_to_goal / 0.1); num_stitch = max(2, num_stitch)
                    stitch_path = np.linspace(last_astar_point, goal_position, num_stitch)
                    global_ref_path = np.vstack([astar_path, stitch_path])
                else: global_ref_path = astar_path

            num_global = global_ref_path.shape[0]
            ref_path = np.zeros((baseline_mpc_params['P'], 3))

            path_distances = np.linalg.norm(np.diff(global_ref_path, axis=0), axis=1)
            cumulative_dist = np.insert(np.cumsum(path_distances), 0, 0)
            for i in range(baseline_mpc_params['P']):
                lookahead_dist = 1.0 * (i + 1) * model_params['dt']
                target_index = np.searchsorted(cumulative_dist, lookahead_dist); target_index = min(target_index, num_global-1)
                ref_path[i, :] = global_ref_path[target_index, :]

            baseline_plan = baseline_optimizer.solve(baseline_state, ref_path, visible_obstacle_means, cost_weights)
            if baseline_plan is not None:
                baseline_state = baseline_plan[1, :]
            baseline_history.append(baseline_state[0:3])
        
        if not riskaware_finished:
            print("  - Planning for Risk-Aware...")
            risk_uav_pos = riskaware_state[0:3]
            
            visible_obstacles_noisy = noisy_perception.detect_obstacles(risk_uav_pos, all_world_obstacles)
            visible_obstacle_means = [obs[0] for obs in visible_obstacles_noisy] # extract means for A*

            ref_planner.update_grid(visible_obstacle_means, risk_aware_mpc_params['safety_distance'])
            astar_path = ref_planner.find_path(risk_uav_pos, goal_position)

            if astar_path is None:
                direction_vector_loop = goal_position - uav_pos
                direction_norm_loop = direction_vector_loop / np.linalg.norm(direction_vector_loop)
                global_ref_path = np.zeros((baseline_mpc_params['P'], 3))
                for i in range(baseline_mpc_params['P']):
                    time_at_step = (i + 1) * model_params['dt']
                    global_ref_path[i, :] = uav_pos + direction_norm_loop * cost_weights['ref_speed'] * time_at_step
            else:
                last_astar_point = astar_path[-1, :]; dist_to_goal = np.linalg.norm(last_astar_point - goal_position)
                if dist_to_goal > 1.0:
                    num_stitch = int(dist_to_goal / 0.1); num_stitch = max(2, num_stitch)
                    stitch_path = np.linspace(last_astar_point, goal_position, num_stitch)
                    global_ref_path = np.vstack([astar_path, stitch_path])
                else: global_ref_path = astar_path

            num_global = global_ref_path.shape[0]
            ref_path = np.zeros((baseline_mpc_params['P'], 3))

            path_distances = np.linalg.norm(np.diff(global_ref_path, axis=0), axis=1)
            cumulative_dist = np.insert(np.cumsum(path_distances), 0, 0)
            for i in range(baseline_mpc_params['P']):
                lookahead_dist = 1.0 * (i + 1) * model_params['dt']
                target_index = np.searchsorted(cumulative_dist, lookahead_dist); target_index = min(target_index, num_global-1)
                ref_path[i, :] = global_ref_path[target_index, :]

            riskaware_plan = riskaware_optimizer.solve(riskaware_state, ref_path, visible_obstacles_noisy, cost_weights)
            if riskaware_plan is not None:
                riskaware_state = riskaware_plan[1, :]
            riskaware_history.append(riskaware_state[0:3])
        
        visualizer.update(baseline_history, riskaware_history, riskaware_plan[:, 0:3] if riskaware_plan is not None else np.array([]))
        time.sleep(model_params['dt'])

        print(baseline_finished)
        if not baseline_finished and np.linalg.norm(baseline_state[0:3] - goal_position) < 0.4:
            baseline_finished = True
            baseline_time = current_time
            print(f"*** Baseline drone finished at {baseline_time:.2f}s ***")
        print(riskaware_finished)
        if not riskaware_finished and np.linalg.norm(riskaware_state[0:3] - goal_position) < 0.4:
            riskaware_finished = True
            riskaware_time = current_time
            print(f"*** Risk-Aware drone finished at {riskaware_time:.2f}s ***")

        if baseline_finished and riskaware_finished:
            break

    print("\n" + "="*50)
    print("---           RACE RESULTS           ---")
    print("="*50)
    print(f"| Method        | Finish Time (s) |")
    print("|---------------|-----------------|")
    print(f"| Baseline      | {baseline_time:<15.2f} |")
    print(f"| Risk-Aware    | {riskaware_time:<15.2f} |")
    print("="*50)
    if riskaware_time < baseline_time:
        improvement = (baseline_time - riskaware_time) / baseline_time * 100
        print(f"Performance Improvement with Risk-Aware MPC: {improvement:.2f}%")

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    race_simulation()
    

    