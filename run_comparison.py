import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm

from mpc_optimizer import MPCOptimizer, NoisyPerceptionModule
from reference_planner import ReferencePlanner
from risk_aware_optimizer import RiskAwareMPCOptimizer
from record_sim import VideoRecorder
import time

class ComparisonVisualizer:
    """A class to handle real-time plotting with a DYNAMIC safety radius"""
    def __init__(self, x_lim, y_lim, goal_position, all_world_obstacles, baseline_safety_radius, risk_aware_params):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.ax.set_aspect('equal', adjustable='box')
        
        self.risk_aware_base_radius = risk_aware_params['safety_distance']
        self.z_score = norm.ppf(1 - risk_aware_params['max_risk'])
        
        self.ax.plot(goal_position[0], goal_position[1], 'g*', markersize=20, label='Goal')
        
        for obs in all_world_obstacles:
            circle = plt.Circle((obs[0], obs[1]), baseline_safety_radius, color='r', alpha=0.15)
            self.ax.add_artist(circle)
            self.ax.plot(obs[0], obs[1], 'rx', markersize=15, label='Obstacle' if 'Obstacle' not in self.ax.get_legend_handles_labels()[1] else "")

        self.baseline_path_plot, = self.ax.plot([], [], 'r-o', markersize=4, label='Baseline Path')
        self.riskaware_path_plot, = self.ax.plot([], [], 'b-o', markersize=4, label='RiskAware Path')
        self.mpc_plan_plot, = self.ax.plot([], [], 'm--', label='RiskAware MPC Plan')
        
        self.dynamic_circles = []

        baseline_safety_proxy = Patch(facecolor='red', alpha=0.15, label='Baseline Fixed Margin')
        riskaware_safety_proxy = Patch(facecolor='blue', alpha=0.3, label='RiskAware Dynamic Margin')

        handles, labels = self.ax.get_legend_handles_labels()
        handles.extend([baseline_safety_proxy, riskaware_safety_proxy])
        self.ax.legend(handles=handles, loc='upper left')

        self.ax.set_title("Comparison: Baseline vs. Risk Aware MPC", fontsize=16)
        self.ax.set_xlabel("X position (m)"); self.ax.set_ylabel("Y position (m)")
        self.ax.set_xlim(x_lim); self.ax.set_ylim(y_lim)
        self.ax.grid(True)

    def update(self, baseline_history, riskaware_history, riskaware_plan, riskaware_pos, visible_obstacles_noisy):
        self.baseline_path_plot.set_data(np.array(baseline_history)[:, 0], np.array(baseline_history)[:, 1])
        self.riskaware_path_plot.set_data(np.array(riskaware_history)[:, 0], np.array(riskaware_history)[:, 1])
        self.mpc_plan_plot.set_data(riskaware_plan[:, 0], riskaware_plan[:, 1])
        
        for circle in self.dynamic_circles:
            circle.remove()
        self.dynamic_circles.clear()

        for obs_mean, obs_cov in visible_obstacles_noisy:
            vec_to_obs = obs_mean - riskaware_pos
            dist_to_obs = np.linalg.norm(vec_to_obs)
            if dist_to_obs < 1e-6: continue
            
            direction_vec = vec_to_obs / dist_to_obs
            projected_variance = direction_vec.T @ obs_cov @ direction_vec
            projected_std_dev = np.sqrt(projected_variance)
            
            required_margin = self.risk_aware_base_radius + self.z_score * projected_std_dev
            
            dynamic_circle = plt.Circle((obs_mean[0], obs_mean[1]), required_margin, color='b', alpha=0.3)
            self.ax.add_artist(dynamic_circle)
            self.dynamic_circles.append(dynamic_circle)

        self.fig.canvas.draw()
        plt.pause(0.01)

def race_simulation():
    print("--- Running Live Comparison: Baseline vs. Risk Aware MPC ---")

    model_params = {'dt': 0.1, 'D_max': np.diag([0.5, 0.5, 0.5])}
    cost_weights = { 'wt': 5.0, 'ws': 1.0, 'wc': 10000.0, 'wj': 1.0, 'alpha': 50.0, 'ref_speed': 1.0 }

    noise_std_dev_lateral = 0.05
    noise_std_dev_depth = 0.2 # 0.30
    baseline_safety_margin = 0.6 + 2.5 * noise_std_dev_depth 
    riskaware_safety_margin = 0.6
    baseline_mpc_params = { 'P': 20, 'v_max': 2.0, 'a_max': 9.81, 'j_max': 1.0, 'safety_distance': baseline_safety_margin}
    risk_aware_mpc_params = {'P': 20, 'v_max': 2.0, 'a_max': 9.81, 'j_max': 1.0, 'safety_distance': riskaware_safety_margin, 'max_risk': 0.01 }
    
    baseline_optimizer = MPCOptimizer(model_params, baseline_mpc_params, cost_weights)
    riskaware_optimizer = RiskAwareMPCOptimizer(model_params, risk_aware_mpc_params, cost_weights)

    goal_position = np.array([10.0, 2.5, 1.0])
    all_world_obstacles = [
        np.array([2.0, 2.0, 1.0]), np.array([4.0, 0.3, 1.0]),
        np.array([6.0, 1.8, 1.0]), np.array([8.0, 1.5, 1.0]),
        np.array([5.0, 3.5, 1.0])
    ]
    #all_world_obstacles = []
    #for x_pos in np.arange(2.0, 9.0, 2):
    #    all_world_obstacles.append(np.array([x_pos, 1.25, 1.0]))  # Top wall
    #    all_world_obstacles.append(np.array([x_pos, -1.25, 1.0])) # Bottom wall

    grid_params = {'width': 120, 'height': 60, 'resolution': 0.1, 'origin_x': -1.0, 'origin_y': -2.0}
    
    sensor_range = 2.0
    noisy_perception = NoisyPerceptionModule(sensor_range=sensor_range, noise_std_dev_lateral=noise_std_dev_lateral,
                                             noise_std_dev_depth=noise_std_dev_depth)
    ref_planner = ReferencePlanner(grid_params)

    baseline_state = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    riskaware_state = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    baseline_history = [baseline_state[0:3]]
    riskaware_history = [riskaware_state[0:3]]

    baseline_time = float('inf')
    riskaware_time = float('inf')
    baseline_control_history = []
    riskaware_control_history = []
    baseline_finished = False
    riskaware_finished = False
    
    visualizer = ComparisonVisualizer(
        x_lim=[-1, 11], y_lim=[-1, 5],
        goal_position=goal_position, all_world_obstacles=all_world_obstacles,
        baseline_safety_radius=baseline_mpc_params['safety_distance'],
        risk_aware_params=risk_aware_mpc_params
    )
    vid_path = "./output/baseline_comparison.mp4"
    video_recorder = VideoRecorder(fig=visualizer.fig, filename=vid_path, fps=15)

    def planning_pipeline(current_state, safety_margin):
        uav_pos = current_state[0:3]
        
        ref_planner.update_grid(all_world_obstacles, safety_margin)
        astar_path = ref_planner.find_path(uav_pos, goal_position)
        
        # Path stitching
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
        
        # Reference path generation
        num_global = global_ref_path.shape[0]
        ref_path = np.zeros((baseline_mpc_params['P'], 3))

        path_distances = np.linalg.norm(np.diff(global_ref_path, axis=0), axis=1)
        cumulative_dist = np.insert(np.cumsum(path_distances), 0, 0)
        for i in range(baseline_mpc_params['P']):
            lookahead_dist = 1.0 * (i + 1) * model_params['dt']
            target_index = np.searchsorted(cumulative_dist, lookahead_dist); target_index = min(target_index, num_global-1)
            ref_path[i, :] = global_ref_path[target_index, :]
        
        return ref_path
    
    # Main Simulation Loop
    num_sim_steps = 150
    for step in range(num_sim_steps):
        current_time = step * model_params['dt']
        print(f"\n--- Step {step+1}/{num_sim_steps} ---")
        print(np.linalg.norm(baseline_state[0:2] - goal_position[0:2]))
        print(np.linalg.norm(riskaware_state[0:2] - goal_position[0:2]))

        if not baseline_finished:
            print("  - Planning for Baseline (Timid)...")
            # Perception
            visible_obstacles_noisy = noisy_perception.detect_obstacles(baseline_state[0:3], all_world_obstacles)
            visible_obstacle_means = [obs[0] for obs in visible_obstacles_noisy]
            # Planning
            baseline_ref_path = planning_pipeline(baseline_state, baseline_mpc_params['safety_distance'])
            baseline_plan = baseline_optimizer.solve(baseline_state, baseline_ref_path, visible_obstacle_means, cost_weights)
            baseline_effort = (baseline_plan[1, 6:9] - baseline_plan[0, 6:9]) / model_params['dt']
            baseline_control_history.append(baseline_effort)
            # Acting
            if baseline_plan is not None:
                baseline_state = baseline_plan[1, :]
            
            if np.linalg.norm(baseline_state[0:2] - goal_position[0:2]) < 0.2:
                baseline_finished = True
                baseline_time = current_time
                print(f"*** Baseline drone finished at {baseline_time:.2f}s ***")

        baseline_history.append(baseline_state[0:3])

        if not riskaware_finished:
            print("  - Planning for Risk-Aware...")
            # Perception
            visible_obstacles_noisy_risk = noisy_perception.detect_obstacles(riskaware_state[0:3], all_world_obstacles)
            # Planning
            riskaware_ref_path = planning_pipeline(riskaware_state, risk_aware_mpc_params['safety_distance'])
            riskaware_plan = riskaware_optimizer.solve(riskaware_state, riskaware_ref_path, visible_obstacles_noisy_risk, cost_weights)
            riskaware_effort = (riskaware_plan[1, 6:9] - riskaware_plan[0, 6:9]) / model_params['dt']
            riskaware_control_history.append(riskaware_effort)
            # Acting
            if riskaware_plan is not None:
                riskaware_state = riskaware_plan[1, :]

            if np.linalg.norm(riskaware_state[0:2] - goal_position[0:2]) < 0.2:
                riskaware_finished = True
                riskaware_time = current_time
                print(f"*** Risk-Aware drone finished at {riskaware_time:.2f}s ***")

        riskaware_history.append(riskaware_state[0:3])

        if 'riskaware_plan' not in locals() or riskaware_plan is None:
            riskaware_plan = np.array([riskaware_state[0:3]] * risk_aware_mpc_params['P'])

        visualizer.update(baseline_history, riskaware_history, riskaware_plan[:, 0:3],
                          riskaware_state[0:3], visible_obstacles_noisy_risk if 'visible_obstacles_noisy_risk' in locals() else [])
        video_recorder.capture_frame()
        time.sleep(model_params['dt'])

        if baseline_finished and riskaware_finished:
            print("\nBoth drones have reached the goal")
            break

    print("\nSimulation finished. Saving video...")
    video_recorder.save_video()

    baseline_path_history = np.array(baseline_history)
    baseline_control_history = np.array(baseline_control_history)
    riskaware_path_history = np.array(riskaware_history)
    riskaware_control_history = np.array(riskaware_control_history)

    baseline_path_segments = np.linalg.norm(np.diff(baseline_path_history, axis=0), axis=1)
    baseline_length = np.sum(baseline_path_segments)
    riskaware_path_segments = np.linalg.norm(np.diff(riskaware_path_history, axis=0), axis=1)
    riskaware_length = np.sum(riskaware_path_segments)

    baseline_control_effort = np.sum(np.linalg.norm(baseline_control_history, axis=1)**2)
    riskaware_control_effort = np.sum(np.linalg.norm(riskaware_control_history, axis=1)**2)

    print("\n" + "="*50)
    print("---           RACE RESULTS           ---")
    print("="*50)
    print(f"| Method        | Finish Time (s) | Path Length (m) | Control Effort")
    print("|---------------|-----------------|-----------------|-----------------|")
    print(f"| Baseline      | {baseline_time:<15.4f} |{baseline_length:<15.4f} |{baseline_control_effort:<15.4f} |")
    print(f"| Risk-Aware    | {riskaware_time:<15.4f} |{riskaware_length:<15.4f} |{riskaware_control_effort:<15.4f} |")
    print("="*50)

    if riskaware_time < baseline_time:
        improvement = (baseline_time - riskaware_time) / baseline_time * 100
        print(f"Performance improvement in time: {improvement:.2f}%")

    if riskaware_length < baseline_length:
        improvement = (baseline_length - riskaware_length) / baseline_length * 100
        print(f"Performance improvement in path length: {improvement:.2f}%")

    if riskaware_control_effort < baseline_control_effort:
        improvement = (baseline_control_effort - riskaware_control_effort) / baseline_control_effort * 100
        print(f"Performance improvement in control effort: {improvement:.2f}%")

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    race_simulation()
    

    