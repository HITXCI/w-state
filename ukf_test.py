import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
import pandas as pd

def move_avg(array, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(array, window, "same")


class UKF:
    def __init__(self, n, m, Q, R, dt=0.01):

        self.n = n
        self.m = m
        self.Q = Q
        self.R = R
        self.dt = dt
        
        # UKF参数
        self.alpha = 0.001
        self.beta = 2
        self.kappa = 0
        self.lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        # 权重计算
        self.Wm = np.zeros(2*n + 1)  # 均值权重
        self.Wc = np.zeros(2*n + 1)  # 协方差权重
        
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2*n + 1):
            self.Wm[i] = 1 / (2 * (n + self.lambda_))
            self.Wc[i] = self.Wm[i]
        

        self.x = np.zeros(n)
        self.P = np.eye(n) * 0.1
        
    def generate_sigma_points(self):

        n = self.n
        lambda_ = self.lambda_
        
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = self.x

        S = cholesky((n + lambda_) * self.P, lower=True)
        
        for i in range(n):
            sigma_points[i + 1] = self.x + S[i]
            sigma_points[n + i + 1] = self.x - S[i]
            
        return sigma_points
    
    def state_transition(self, x, dt):

        F = np.array([[1, dt],
                      [0, 1]])
        return F @ x
    
    def observation_model(self, x):

        H = np.array([[1, 0]])
        return H @ x
    
    def predict(self):

        sigma_points = self.generate_sigma_points()
        

        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(len(sigma_points)):
            sigma_points_pred[i] = self.state_transition(sigma_points[i], self.dt)

        self.x_pred = np.zeros(self.n)
        for i in range(len(sigma_points_pred)):
            self.x_pred += self.Wm[i] * sigma_points_pred[i]
        

        self.P_pred = np.zeros((self.n, self.n))
        for i in range(len(sigma_points_pred)):
            diff = sigma_points_pred[i] - self.x_pred
            self.P_pred += self.Wc[i] * np.outer(diff, diff)
        self.P_pred += self.Q

        self.sigma_points_pred = sigma_points_pred
        
    def update(self, z):

        sigma_points_obs = np.zeros((len(self.sigma_points_pred), self.m))
        for i in range(len(self.sigma_points_pred)):
            sigma_points_obs[i] = self.observation_model(self.sigma_points_pred[i])

        z_pred = np.zeros(self.m)
        for i in range(len(sigma_points_obs)):
            z_pred += self.Wm[i] * sigma_points_obs[i]

        Pzz = np.zeros((self.m, self.m))
        Pxz = np.zeros((self.n, self.m))
        
        for i in range(len(sigma_points_obs)):

            diff_z = sigma_points_obs[i] - z_pred
            Pzz += self.Wc[i] * np.outer(diff_z, diff_z)
            

            diff_x = self.sigma_points_pred[i] - self.x_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
        
        Pzz += self.R

        K = Pxz @ np.linalg.inv(Pzz)

        self.x = self.x_pred + K @ (z - z_pred)
        self.P = self.P_pred - K @ Pzz @ K.T
        
        return self.x

def add_measurement_noise(data, noise_std=0.5):

    noise = np.random.normal(0, noise_std, len(data))
    return data + noise

def simulate_vehicle_data():

    t = np.linspace(0, 10, 1000)

    true_beta = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
    return t, true_beta

def main():

    np.random.seed(42)

    cfc_path = "/home/rrc/xy_projects/LLN/carsim/output/all_preds.csv"
    cfc = pd.read_csv(cfc_path)
    t = cfc['time'].values.astype(np.float32)
    true_beta = cfc['gt_beta'].values.astype(np.float32) * 180 / np.pi
    t = t[:100]
    true_beta = true_beta[:100]
    dt = t[1] - t[0]   

    measured_beta = add_measurement_noise(true_beta, noise_std=18)

    n = 2  
    m = 1  
    

    Q = np.diag([0.01, 0.1]) 

    R = np.array([[0.09]]) 
    
    ukf = UKF(n, m, Q, R, dt)
    

    ukf.x = np.array([measured_beta[0], 0]) 
    

    estimated_beta = np.zeros(len(t))
    estimated_beta_rate = np.zeros(len(t))

    for i in range(len(t)):

        ukf.predict()
        
        estimated_state = ukf.update(np.array([measured_beta[i]]))
        
        estimated_beta[i] = estimated_state[0]
        estimated_beta_rate[i] = estimated_state[1]
    

    measurement_rmse = np.sqrt(np.mean((measured_beta - true_beta)**2))
    ukf_rmse = np.sqrt(np.mean((estimated_beta - true_beta)**2))
    
    print(f"测量值RMSE: {measurement_rmse:.4f} deg")
    print(f"UKF估计RMSE: {ukf_rmse:.4f} deg")
    print(f"改进: {(measurement_rmse - ukf_rmse)/measurement_rmse*100:.2f}%")
    
    estimated_beta = move_avg(estimated_beta, 3)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t[:300], true_beta[:300], 'g-', linewidth=2, label='gt-beta')
    # plt.plot(t[:300], measured_beta[:300], 'r--', alpha=0.6, label='measure')
    plt.plot(t[:300], estimated_beta[:300], 'b-', linewidth=1.5, label='UKF-est')
    plt.ylabel('beta (deg)')
    plt.legend()
    plt.grid(True)

    header = "time, beta_output"
    data_ukf = np.stack((t, estimated_beta), axis=1)
    np.savetxt("./output_2_beta.csv", data_ukf, delimiter=',', header=header)
    
    plt.subplot(2, 1, 2)
    plt.plot(t[:300], estimated_beta_rate[:300], 'purple', linewidth=1.5, label='beta-rate_est')
    plt.ylabel('beta-rate (deg/s)')
    plt.xlabel('t (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    measurement_error = np.abs(measured_beta - true_beta)
    ukf_error = np.abs(estimated_beta - true_beta)
    
    plt.plot(t, measurement_error, 'r--', alpha=0.7, label='measure-err')
    plt.plot(t, ukf_error, 'b-', label='ukf-est-err')
    plt.ylabel('err (deg)')
    plt.xlabel('t (s)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()