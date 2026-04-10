import numpy as np
import matplotlib.pyplot as plt

def solve_md_refined(temp_c, crowding=False):
    # Convert Celsius to Sim Units for the math
    temp = (temp_c + 273.15) / 100 
    N, dt, steps = 12, 0.01, 8000
    pos = np.zeros((N, 3))
    pos[:, 0] = np.linspace(0, 3, N)
    vel = np.random.normal(0, np.sqrt(temp), (N, 3))
    
    rg_history = []
    for _ in range(steps):
        forces = np.zeros((N, 3))
        # 1. Harmonic Bonds
        for i in range(N-1):
            r = pos[i+1] - pos[i]
            dist = np.linalg.norm(r)
            forces[i] += 150 * (dist - 1.0) * (r / dist)
            forces[i+1] -= 150 * (dist - 1.0) * (r / dist)
        
        # 2. Stronger Native Contacts (Essential to see melting)
        for i in range(N-3):
            r = pos[i+3] - pos[i]
            dist = np.linalg.norm(r)
            if dist < 3.0:
                forces[i] += 15.0 * (dist - 1.2) * (r / dist) # Stronger well
                forces[i+3] -= 15.0 * (dist - 1.2) * (r / dist)

        # 3. Crowding (Steric interactions)
        if crowding:
            box = 2.8 # Tighten the box slightly
            for i in range(N):
                d = np.linalg.norm(pos[i])
                if d > box: forces[i] -= 30.0 * (d - box) * (pos[i]/d)

        vel = (vel + forces * dt) * 0.96 # Thermostat
        pos += vel * dt
        rg = np.sqrt(np.mean(np.sum((pos - np.mean(pos, axis=0))**2, axis=1)))
        rg_history.append(rg)
    return np.mean(rg_history[4000:])

# Generate Temperature range in Celsius (like the papers)
temps_c = np.linspace(10, 90, 20)
rg_dilute = np.array([solve_md_refined(t, False) for t in temps_c])
rg_crowded = np.array([solve_md_refined(t, True) for t in temps_c])

# Normalize to get "Fraction Unfolded" (0 to 1)
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

f_u_dilute = normalize(rg_dilute)
f_u_crowded = normalize(rg_crowded)

# Plotting to match Nature Comm / Protein Science style
plt.figure(figsize=(7, 5))
plt.plot(temps_c, f_u_dilute, 'ko-', label='Dilute Buffer', markersize=4)
plt.plot(temps_c, f_u_crowded, 'ro-', label='Crowded (Ficoll-like)', markersize=4)
plt.axhline(0.5, color='gray', linestyle='--') # Tm line
plt.xlabel('Temperature (°C)')
plt.ylabel('Normalized Signal (Fraction Unfolded)')
plt.title('Protein Stability: MD Simulation vs Paper Trend')
plt.legend()
plt.grid(alpha=0.3)
plt.show()