import time
import numpy as np
import matplotlib.pyplot as plt

# state = [x_pos, y_pos]
num_data = 10
ground_truth_x = np.linspace(0, 10, num=num_data + 1)
ground_truth_y = ground_truth_x.copy() # x = y

# Simulate dynamics
x_0, y_0 = 0, 0
xs, ys = [0], [0]
dynamics_noise_x_var = 0.3
dynamics_noise_y_var = 0.3
for _ in range(10):
    v_x, v_y = 1.0, 1.0
    noise_x = np.random.normal(loc=0.0, scale=dynamics_noise_x_var)
    noise_y = np.random.normal(loc=0.0, scale=dynamics_noise_y_var)
    new_x = xs[-1] + v_x + noise_x
    new_y = ys[-1] + v_y + noise_y
    xs.append(new_x)
    ys.append(new_y)
    
# Simulate measurements
measurement_noise_x_var = 0.75
measurement_noise_y_var = 0.6
noise_x = np.random.normal(loc=0.0, scale=measurement_noise_x_var, size=num_data-1)
noise_y = np.random.normal(loc=0.0, scale=measurement_noise_y_var, size=num_data-1)
measurement_x = np.linspace(1, 10, num=num_data-1) + noise_x
measurement_y = np.linspace(1, 10, num=num_data-1) + noise_y

# Compare ground truth and measurements
plt.plot(ground_truth_x, ground_truth_y)
plt.plot(measurement_x, measurement_y)
plt.plot(xs, ys)
plt.xlabel('x position')
plt.ylabel('y position')
plt.legend(['ground truth', 'measurements', 'dynamics'])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


def predict(A, B, Q, mu_t, u_t, Sigma_t):
    predicted_mu = A @ mu_t + B @ u_t
    predicted_Sigma = A @ Sigma_t @ A.T + Q
    return predicted_mu, predicted_Sigma

def update(H, R, z, predicted_mu, predicted_Sigma):
    residual_mean = z - H @ predicted_mu
    residual_covariance = H @ predicted_Sigma @ H.T + R
    kalman_gain = predicted_Sigma @ H.T @ np.linalg.inv(residual_covariance)
    updated_mu = predicted_mu + kalman_gain @ residual_mean
    updated_Sigma = predicted_Sigma - kalman_gain @ H @ predicted_Sigma
    return updated_mu, updated_Sigma

# Initialize the problem
mu_0 = np.array([0, 0])
Sigma_0 = np.array([[0.1, 0],
                     [0, 0.1]]) # We're pretty certain with mu_0
A = np.array([[1, 0],
              [0, 1]])
B = np.array([[1, 0],
              [0, 1]])
Q = np.array([[0.3, 0],
              [0, 0.3]])
H = np.array([[1, 0],
              [0, 1]])
R = np.array([[measurement_noise_x_var, 0],
              [0, measurement_noise_y_var]])

# Initialize empty lists for mus and measurements for plotting
measurements = []
filtered_mus = []

# Run KF for each time step
mu_current = mu_0.copy()
Sigma_current = Sigma_0.copy()
for i in range(num_data-1):
    u_t = np.array([1, 1])
    
    # Predict step
    predicted_mu, predicted_Sigma = predict(A, B, Q, 
                                            mu_current, u_t, 
                                            Sigma_current)
    
    # Get measurement (irl, get this from our sensor)
    measurement_noise_x = np.random.normal(loc=0.0, scale=measurement_noise_x_var)
    measurement_noise_y = np.random.normal(loc=0.0, scale=measurement_noise_y_var)
    measurement_x_new = ground_truth_x[i+1] + measurement_noise_x
    measurement_y_new = ground_truth_x[i+1] + measurement_noise_y
    z = np.array([measurement_x_new, measurement_y_new])
    
    # The rest of update step
    mu_current, Sigma_current = update(H, R, z, 
                                   predicted_mu, 
                                   predicted_Sigma)
    
    # Store measurements and mu_current so we can plot it later
    measurements.append([measurement_x_new, measurement_y_new])
    filtered_mus.append(mu_current)

# Just for plotting purposes, convert the lists to array 
measurements = np.array(measurements)
filtered_mus = np.array(filtered_mus) 

# Let's plot the results

plt.plot(ground_truth_x, ground_truth_y)
plt.plot(measurements[:,0], measurements[:,1])
plt.plot(xs, ys)
plt.plot(filtered_mus[:,0], filtered_mus[:,1])
plt.xlabel('x position')
plt.ylabel('y position')
plt.legend(['ground truth', 'measurements', 'dynamics', 'KF'])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()