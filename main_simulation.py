import pybullet as p
import time
import numpy as np
from mpc_optimizer import MPCOptimizer
import matplotlib.pyplot as plt

def setup_simulation():
    """Initializes PyBullet and creates the simulation environment."""
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, 0) # We handle all physics, so disable gravity
    
    # Create the UAV object (a simple sphere)
    uav_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[0.1, 0.2, 0.8, 1])
    # Set mass to 0 to make it a purely kinematic object
    uav_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=uav_visual_shape_id, basePosition=[0,0,1])
    
    # Create obstacle objects
    obstacle_positions = [[1.5, 0.5, 1.0], [3.0, 1.5, 1.0]]
    for pos in obstacle_positions:
        obs_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=0.6, length=2, rgbaColor=[1, 0.2, 0.2, 0.8])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=obs_visual_shape_id, basePosition=pos)
    
    print("PyBullet visualizer environment created")
    return uav_id, obstacle_positions

def plot_results(global_ref_path, path_history, obstacles, mpc_params, goal_position):
    """Plots the final executed path."""
    fig, ax = plt.subplots(figsize=(10, 8))
    path_history = np.array(path_history)

    ax.plot(global_ref_path[:, 0], global_ref_path[:, 1], 'g--', label='Reference Path')
    ax.plot(path_history[:, 0], path_history[:, 1], 'b-o', markersize=4, label='Executed Path')
    ax.plot(path_history[0, 0], path_history[0, 1], 'ks', markersize=10, label='Start')
    ax.plot(goal_position[0], goal_position[1], 'g*', markersize=15, label='Goal')
    
    safety_radius = mpc_params['safety_distance']
    obs = obstacles[0]
    circle = plt.Circle((obs[0], obs[1]), safety_radius, color='r', alpha=0.3)
    ax.add_artist(circle)
    ax.plot(obs[0], obs[1], 'rx', markersize=10, label='Obstacle')
    for obs in obstacles[1:]:
        circle = plt.Circle((obs[0], obs[1]), safety_radius, color='r', alpha=0.3)
        ax.add_artist(circle)
        ax.plot(obs[0], obs[1], 'rx', markersize=10)

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_title("MPC Simulation Results")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.show()

def run_simulation():
    # Initialization 
    uav_id, obstacle_positions = setup_simulation()
    
    model_params = {'dt': 0.1, 'D_max': np.diag([0.1, 0.1, 0.1])}
    mpc_params = { 'P': 20, 'v_max': 2.0, 'a_max': 3.0, 'j_max': 2.0, 'safety_distance': 0.6 }
    cost_weights = { 'wt': 5.0, 'ws': 1.0, 'wc': 10000.0, 'wj': 0.5, 'alpha': 50.0, 'ref_speed': 1.0 }
    
    optimizer = MPCOptimizer(model_params, mpc_params, cost_weights)
    
    # State and Goal Setup
    current_state = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    goal_position = np.array([4.0, 2.0, 1.0])
    path_history = [current_state[0:3]]

    num_ref_points = int(np.linalg.norm(goal_position - current_state[0:3]) / (cost_weights['ref_speed'] * model_params['dt']))
    global_ref_path = np.zeros((num_ref_points, 3))
    direction_vector = goal_position - current_state[0:3]
    direction_norm = direction_vector / np.linalg.norm(direction_vector)
    for i in range(num_ref_points):
        time_at_step = (i + 1) * model_params['dt']
        global_ref_path[i, :] = current_state[0:3] + direction_norm * 1.0 * time_at_step

    # Simulation Loop
    num_sim_steps = 100
    for step in range(num_sim_steps):
        # Check if the user has closed the window
        if p.getConnectionInfo()['isConnected'] == 0: break
        print(f"\n--- Simulation Step {step + 1}/{num_sim_steps} ---")
        if np.linalg.norm(current_state[0:3] - goal_position) < 0.2:
            print("Goal reached!")
            break
        
        # GENERATE REFERENCE
        direction_vector_loop = goal_position - current_state[0:3]
        direction_norm_loop = direction_vector_loop / np.linalg.norm(direction_vector_loop)
        ref_path = np.zeros((mpc_params['P'], 3))
        for i in range(mpc_params['P']):
            time_at_step = (i + 1) * model_params['dt']
            ref_path[i, :] = current_state[0:3] + direction_norm_loop * cost_weights['ref_speed'] * time_at_step
            
        # PLAN
        optimized_trajectory = optimizer.solve(current_state, ref_path, obstacle_positions, cost_weights)
        if optimized_trajectory is None:
            print("Solver failed to find a solution. Halting.")
            break
            
        # ACT
        current_state = optimized_trajectory[1, :]
        path_history.append(current_state[0:3])
        
        # VISUALIZE
        p.resetBasePositionAndOrientation(uav_id, current_state[0:3], p.getQuaternionFromEuler([0,0,0]))
        p.addUserDebugLine(current_state[0:3], goal_position, [0,1,0], 2, lifeTime=0.2)
        for i in range(mpc_params['P']):
            p.addUserDebugLine(optimized_trajectory[i, 0:3], optimized_trajectory[i+1, 0:3], [0,0,1], 2, lifeTime=0.2)
        
        time.sleep(model_params['dt'])

    print("\nSimulation finished.")
    p.disconnect()
    
    # Plot final results
    plot_results(global_ref_path=global_ref_path, path_history=path_history, obstacles=obstacle_positions,
                  mpc_params=mpc_params, goal_position=goal_position)

if __name__ == '__main__':
    run_simulation()