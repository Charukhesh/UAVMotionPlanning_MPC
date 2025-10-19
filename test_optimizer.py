from mpc_optimizer import MPCOptimizer
import matplotlib.pyplot as plt
import numpy as np

def plot_results(global_ref_path, optimized_trajectory, obstacles, mpc_params):
    """
    Plots the planned trajectory in 2D (top-down view)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get the position data from the full state trajectory
    opt_pos = optimized_trajectory[:, 0:3]
    
    # Plot reference path (the 'suggestion')
    ax.plot(global_ref_path[:, 0], global_ref_path[:, 1], 'g--', label='Reference Path')
    
    # Plot the optimized MPC trajectory (the 'plan')
    ax.plot(opt_pos[:, 0], opt_pos[:, 1], 'b-o', markersize=4, label='MPC Trajectory')
    
    # Plot the starting position
    ax.plot(opt_pos[0, 0], opt_pos[0, 1], 'ks', markersize=10, label='Start')

    safety_radius = mpc_params['safety_distance']
    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), safety_radius, color='r', alpha=0.3)
        ax.add_artist(circle)
        ax.text(obs[0], obs[1], 'Obs', ha='center', va='center', color='white', fontsize=10)

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("MPC Trajectory Optimization")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.show()

def run_optimizer_test():
    """
    Sets up a test case and runs the MPC optimizer once.
    """
    print("--- Running MPC Optimizer Test ---")

    # Define Parameters
    model_params = {
        'dt': 0.1,  # Time step (seconds)
        'D_max': np.diag([0.5, 0.5, 0.5]) # Max drag coefficients
    }

    mpc_params = {
        'P': 20,                # Prediction horizon
        'v_max': 2.0,           # Max velocity (m/s)
        'a_max': 9.81,          # Max acceleration (m/s^2)
        'j_max': 1.0,           # Max jerk (m/s^3)    
        'safety_distance': 0.6  # Safety distance (m)
    }

    # These weights are crucial for tuning the behavior.
    # Start with tracking and jerk penalties, then add others.
    cost_weights = {
        'wt': 5.0,        # Tracking weight
        'ws': 1.0,        # Speed weight
        'wc': 5000.0,    # Collision weight
        'wj': 0.5,        # Jerk weight (to encourage smoothness)
        'alpha': 50.0,    # Safety sharpness for collision cost 
        'ref_speed': 1.0  # Reference speed (m/s)
    }

    # Instantiate the Optimizer
    optimizer = MPCOptimizer(model_params, mpc_params, cost_weights)
    print(f"Optimizer initialized with prediction horizon P = {optimizer.P}")

    # Defining the Test Scenario
    # Initial state: [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z]
    # Starting at the origin, with a small forward velocity.
    current_state = np.array([
        0.0, 0.0, 1.0,  # Position (p) at (0, 0, 1m)
        0.5, 0.0, 0.0,  # Velocity (v) at 0.5 m/s in x
        0.0, 0.0, 0.0   # Acceleration (a) is zero
    ])

    goal_position = np.array([3.0, 1.5, 1.0])
    direction_vector = goal_position - current_state[0:3]
    direction_vector /= np.linalg.norm(direction_vector)

    ref_horizon_time = 5.0
    ref_points = int(ref_horizon_time / model_params['dt'])
    global_ref_path = np.zeros((ref_points, 3))

    for i in range(ref_points):
        # Position = start + direction * speed * time
        time_at_step = (i+1) * model_params['dt']
        global_ref_path[i, :] = current_state[0:3] + direction_vector * cost_weights['ref_speed'] * time_at_step

    ref_path = global_ref_path[0:mpc_params['P'], :]

    # Obstacles: Move the obstacle directly in front of the drone
    obstacles = [
        np.array([1.5, 0.5, 1.0]),  
    ]

    print(f"\nScenario defined:")
    print(f"Current State (p,v,a): {np.round(current_state, 2)}")
    print(f"Number of Obstacles: {len(obstacles)}")
    print(f"Obstacle Position: {obstacles[0]}")

    # Calling the Solver
    print("\nCalling the optimizer's solve method...")
    optimized_trajectory = optimizer.solve(current_state, ref_path, obstacles, cost_weights)
    
    if optimized_trajectory is not None:
        print("\nPlotting the results...")
        plot_results(global_ref_path, optimized_trajectory, obstacles, mpc_params)
        print("\nMPC Optimizer Test Completed.")

if __name__ == '__main__':
    run_optimizer_test()