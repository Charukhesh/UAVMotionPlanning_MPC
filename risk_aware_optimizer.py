import numpy as np
from scipy.stats import norm
from mpc_optimizer import MPCOptimizer

class RiskAwareMPCOptimizer(MPCOptimizer):
    def __init__(self, model_params, mpc_params, cost_weights):
        super().__init__(model_params, mpc_params, cost_weights)
        self.max_risk = mpc_params['max_risk'] # max collision probability
        self.z_score = norm.ppf(1 - self.max_risk)

    def objective_function(self, u_flat, cost_weights, current_state, ref_path, obstacles_with_uncertainty):
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
        
        # NEW: CHANCE-CONSTRAINED COLLISION COST
        cost_c = 0.0
        X = self._rollout_trajectory(current_state, u_flat.reshape(self.P, self.nu))
        P_traj = X[1:, 0:3]

        for i in range(self.P):
            p_current = P_traj[i, :]
            for obs_mean, obs_cov in obstacles_with_uncertainty:
                # Mahalanobis distance as a proxy for collision probability
                # The uncertainty along the vector from drone to obstacle
                vec_to_obs = obs_mean - p_current
                dist_to_obs = np.linalg.norm(vec_to_obs)
                
                if dist_to_obs < 1e-6: continue
                direction_vec = vec_to_obs / dist_to_obs
                
                # Project the full 3x3 covariance matrix onto the 3D direction vector
                # v^T * C * v
                projected_variance = direction_vec.T @ obs_cov @ direction_vec
                projected_std_dev = np.sqrt(projected_variance)
                
                # The required safety margin is our base radius + uncertainty margin
                required_margin = self.safety_distance + self.z_score * projected_std_dev
                
                # Using the quadratic penalty with the DYNAMIC margin
                violation = required_margin - dist_to_obs
                if violation > 0:
                    cost_c += violation**2
        
        total_cost = (self.wt * cost_t + self.ws * cost_s + 
                      self.wj * cost_j + self.wc * cost_c)
        return total_cost