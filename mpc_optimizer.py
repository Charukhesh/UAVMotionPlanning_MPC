import numpy as np
from scipy.optimize import minimize

def get_uav_model(dt, D_max=np.diag([0.5, 0.5, 0.5])):
    """
    Defines the discrete-time linear state-space model for the UAV.
    State x = [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z].T (9x1)
    Control u = [j_x, j_y, j_z].T (3x1) (jerk)
    x_{k+1} = A * x_k + B * u_k
    """
    # State transition matrix A (9x9)
    A = np.eye(9)
    A[0:3, 3:6] = dt * np.eye(3)
    A[3:6, 6:9] = dt * np.eye(3)
    A[3:6, 3:6] -= dt * D_max # Drag term

    # Control matrix B (9x3)
    B = np.zeros((9, 3))
    B[6:9, 0:3] = dt * np.eye(3)
    
    return A, B

class MPCOptimizer:
    def __init__(self, model_params, mpc_params, cost_weights):
        self.A, self.B = get_uav_model(model_params['dt'], model_params['D_max'])
        
        self.P = mpc_params['P']  # Prediction horizon (e.g., 20 steps)
        self.dt = model_params['dt']
        
        # State and control dimensions
        self.nx = self.A.shape[1] # Number of states (9)
        self.nu = self.B.shape[1] # Number of controls (3)
        
        # Cost function weights
        self.wt = cost_weights['wt'] # Tracking
        self.ws = cost_weights['ws'] # Speed
        self.wc = cost_weights['wc'] # Collision
        self.wj = cost_weights['wj'] # Jerk

        self.ref_speed = cost_weights['ref_speed'] # Reference speed
        self.alpha = cost_weights['alpha'] # Sharpness for collision cost
        self.safety_distance = mpc_params['safety_distance'] # Safety distance for collision cost

        # Physical constraints
        self.v_max = mpc_params['v_max']
        self.a_max = mpc_params['a_max']
        self.j_max = mpc_params['j_max']

    def solve(self, current_state, ref_path, obstacles, cost_weights):
        """
        Solves the nonlinear optimization problem to find the optimal control sequence.
        """
        u_initial_guess = np.zeros(self.P * self.nu)

        # Objective function
        obj_func = lambda u: self.objective_function(u, cost_weights, current_state, ref_path, obstacles)

        # Bounds
        jerk_bounds = (-self.j_max, self.j_max)
        bounds = [jerk_bounds] * (self.P * self.nu)

        # Constraints
        vel_constraint_func = lambda u: self._velocity_constraint(u, current_state)
        accel_constraint_func = lambda u: self._acceleration_constraint(u, current_state)

        constraints = [
            {'type': 'ineq', 'fun': vel_constraint_func},
            {'type': 'ineq', 'fun': accel_constraint_func}
        ]

        # Solver
        print("Calling SLSQP solver...")
        result = minimize(
            obj_func,
            u_initial_guess,
            method='SLSQP', # Sequential Least Squares Programming
            bounds=bounds,
            constraints=constraints,
            options={'disp': True, 'maxiter': 100} # 'disp': True shows solver output
        )
        print("Solver finished.")

        # Results
        if result.success:
            optimized_u = result.x.reshape(self.P, self.nu)
            optimized_trajectory = self._rollout_trajectory(current_state, optimized_u)
            return optimized_trajectory
        else:
            print("!!! Optimization failed !!!")
            print(result.message)
            # If it fails, return the trajectory based on the initial (zero jerk) guess
            # This is a basic safety fallback
            return self._rollout_trajectory(current_state, u_initial_guess.reshape(self.P, self.nu))

    def objective_function(self, u_flat, cost_weights, current_state, ref_path, obstacles):
        # Reshape the flat control vector into a sequence of control inputs
        U = u_flat.reshape(self.P, self.nu)
        
        # Predict the trajectory based on this control sequence
        X = self._rollout_trajectory(current_state, U)
        
        # Unpack states
        P_traj = X[:, 0:3]  # Position part of the trajectory
        V_traj = X[:, 3:6]  # Velocity part of the trajectory
        
        # 1. Tracking Cost (Jt) - Eq. 4
        # Note: ref_path should have P points
        cost_t = np.sum((P_traj[1:, :] - ref_path)**2)
        
        # 2. Speed Cost (Js) - Eq. 5
        ref_speed = cost_weights['ref_speed'] # From paper
        speed_errors = np.linalg.norm(V_traj[1:, :], axis=1)**2 - ref_speed**2
        cost_s = np.sum(speed_errors**2)
        
        # 3. Jerk Penalty (Jj) - Eq. 7
        cost_j = np.sum(U**2)
        
        # 4. Collision Cost (Jc) - Eq. 6 (Simplified for now)
        cost_c = 0.0
        for i in range(1, self.P + 1):
            p_current = P_traj[i, :]
            for obs_pos in obstacles:
                dist_to_obs = np.linalg.norm(p_current - obs_pos)
                # Using the logistic function from the paper
                cost_c += 1.0 / (1.0 + np.exp(self.alpha * (dist_to_obs - self.safety_distance)))
                
        # Total Weighted Cost 
        total_cost = (self.wt * cost_t +
                    self.ws * cost_s +
                    self.wj * cost_j +
                    self.wc * cost_c)
                    
        return total_cost

    def _rollout_trajectory(self, initial_state, control_sequence_U):
        """
        Predicts the future trajectory given an initial state and a control sequence.
        """
        trajectory = np.zeros((self.P + 1, self.nx))
        trajectory[0, :] = initial_state
        
        for i in range(self.P):
            trajectory[i+1, :] = self.A @ trajectory[i, :] + self.B @ control_sequence_U[i, :]
            
        return trajectory

    def _velocity_constraint(self, u_flat, current_state):
        """ Constraint: v_max - ||v_i|| >= 0 """
        U = u_flat.reshape(self.P, self.nu)
        X = self._rollout_trajectory(current_state, U)
        V_traj = X[1:, 3:6]  # Velocities at steps 1 to P
        
        # Calculate the magnitude of each velocity vector
        velocities_mag = np.linalg.norm(V_traj, axis=1)
        
        # The constraint is satisfied if v_max - velocity_mag is non-negative
        return self.v_max - velocities_mag

    def _acceleration_constraint(self, u_flat, current_state):
        """ Constraint: a_max - ||a_i|| >= 0 """
        U = u_flat.reshape(self.P, self.nu)
        X = self._rollout_trajectory(current_state, U)
        A_traj = X[1:, 6:9]  # Accelerations at steps 1 to P
        
        # Calculate the magnitude of each acceleration vector
        accelerations_mag = np.linalg.norm(A_traj, axis=1)
        
        # The constraint is satisfied if a_max - acceleration_mag is non-negative
        return self.a_max - accelerations_mag 


