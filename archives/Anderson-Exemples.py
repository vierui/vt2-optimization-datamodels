import numpy as np

U_k = 0.984  # voltage magn @ k (p.u.)
U_m = 0.962  # voltage magn @ m (p.u.)
theta_km_deg = 10  # angle diff(deg)
x_km = 0.0175  # reactance (p.u.)

# Convert deg angle to radians
theta_km_rad = np.radians(theta_km_deg)

#Active Power
# Sensitivity of P_km with respect to angle
dP_dtheta = (U_k * U_m * np.cos(theta_km_rad)) / x_km
print(f"Sensitivity of Pkm with respect to theta: {dP_dtheta:.2f}")
# Sensitivity of active power P_km with respect to voltage
dP_dU = (U_m * np.sin(theta_km_rad)) / x_km
print(f"Sensitivity of Pkm with respect to Uk: {dP_dU:.2f}")

#Reactive Power
# Q_km Sensitivity with respect to angle 
dQ_dtheta = (U_k * U_m * np.sin(theta_km_rad)) / x_km
print(f"Sensitivity of Qkm with respect to theta: {dQ_dtheta:.2f}")
# Q_km Sensitivity with respect to voltage
dQ_dU = (2 * U_k - U_m * np.cos(theta_km_rad)) / x_km
print(f"Sensitivity of Qkm with respect to Uk: {dQ_dU:.2f}")
