import numpy as np
import matplotlib.pyplot as plt

def run_stabilized_md(temp_c, crowding=False):
    # Simulating 20,000 steps for better averaging/equilibrium
    temp = (temp_c + 273.15) / 120 
    N, dt, steps = 14, 0.01, 20000 
    pos = np.zeros((N, 3))
    pos[:, 0] = np.linspace(0, 2, N)
    vel = np.random.normal(0, np.sqrt(temp), (N, 3))
    
    rg_list = []
    for s in range(steps):
        forces = np.zeros((N, 3))
        # 1. Stiff Bonds
        for i in range(N-1):
            r = pos[i+1] - pos[i]
            d = np.linalg.norm(r)
            forces[i] += 200 * (d - 1.0) * (r / d)
            forces[i+1] -= 200 * (d - 1.0) * (r / d)
        
        # 2. Native Contacts (The "Folding Glue")
        for i in range(N-4):
            r = pos[i+4] - pos[i]
            d = np.linalg.norm(r)
            if d < 3.5: # Attractive range
                forces[i] += 25.0 * (d - 1.2) * (r / d)
                forces[i+4] -= 25.0 * (d - 1.2) * (r / d)

        # 3. Crowding (Steric Constraint)
        if crowding:
            box = 2.5 
            for i in range(N):
                d_norm = np.linalg.norm(pos[i])
                if d_norm > box: forces[i] -= 50.0 * (d_norm - box) * (pos[i]/d_norm)

        vel = (vel + forces * dt) * 0.95 
        pos += vel * dt
        if s > 10000: # Only average the second half (Equilibrium)
            rg_list.append(np.sqrt(np.mean(np.sum((pos - np.mean(pos, 0))**2, 1))))
    return np.mean(rg_list)

# Run Simulation
temps_c = np.linspace(5, 95, 20)
rg_dilute = np.array([run_stabilized_md(t, False) for t in temps_c])
rg_crowded = np.array([run_stabilized_md(t, True) for t in temps_c])

# Normalization
f_u_dilute = (rg_dilute - rg_dilute.min()) / (rg_dilute.max() - rg_dilute.min())
f_u_crowded = (rg_crowded - rg_crowded.min()) / (rg_crowded.max() - rg_crowded.min())

# Generate Multi-Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Physical Interpretation
ax1.plot(temps_c, rg_dilute, 'ko-', label='Dilute (Large Volume)')
ax1.plot(temps_c, rg_crowded, 'ro-', label='Crowded (Restricted Volume)')
ax1.set_ylabel('Radius of Gyration (Sim Units)')
ax1.set_title('Physical Protein Size')
ax1.legend()

# Panel 2: Paper-Style Interpretation
ax2.plot(temps_c, f_u_dilute, 'ks-', label='Dilute Buffer')
ax2.plot(temps_c, f_u_crowded, 'rs-', label='Crowded (Cell-like)')
ax2.axhline(0.5, color='blue', linestyle='--', label='Tm Point')
ax2.set_ylabel('Normalized Signal (Fraction Unfolded)')
ax2.set_title('Melting Transition (Paper Style)')
ax2.legend()

plt.tight_layout()
plt.show()