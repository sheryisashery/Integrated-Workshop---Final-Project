import numpy as np
import matplotlib.pyplot as plt

def solve_md(temp, crowding=False):
    # Constants & Parameters
    N = 10 # Beads
    dt = 0.01
    steps = 5000
    pos = np.zeros((N, 3))
    pos[:, 0] = np.arange(N) # Linear start
    vel = np.random.normal(0, np.sqrt(temp), (N, 3))
    
    # Simple Go-like potential + Excluded Volume
    rg_history = []
    for _ in range(steps):
        forces = np.zeros((N, 3))
        # 1. Harmonic Bonds (Connect beads)
        for i in range(N-1):
            r = pos[i+1] - pos[i]
            dist = np.linalg.norm(r)
            forces[i] += 100 * (dist - 1.0) * (r / dist)
            forces[i+1] -= 100 * (dist - 1.0) * (r / dist)
        
        # 2. Native Contacts (Folding Drive)
        for i in range(N-2):
            r = pos[i+2] - pos[i]
            dist = np.linalg.norm(r)
            if dist < 2.5: # Attractive well
                forces[i] += 5.0 * (dist - 1.5) * (r / dist)
                forces[i+2] -= 5.0 * (dist - 1.5) * (r / dist)

        # 3. Crowding Force (Repulsion from "Impurities")
        if crowding:
            # Simple soft-wall repulsion at boundaries to mimic tight space
            box_limit = 3.0
            for i in range(N):
                dist_from_center = np.linalg.norm(pos[i])
                if dist_from_center > box_limit:
                    forces[i] -= 20.0 * (dist_from_center - box_limit) * (pos[i]/dist_from_center)

        # Integration (Verlet-like) & Thermostat
        vel += forces * dt
        vel *= 0.95 # Simple damping to maintain temperature
        pos += vel * dt
        
        # Record Radius of Gyration
        rg = np.sqrt(np.mean(np.sum((pos - np.mean(pos, axis=0))**2, axis=1)))
        rg_history.append(rg)
        
    return np.mean(rg_history[steps//2:]) # Return average size

# Run simulation across temperature range
temps = np.linspace(0.1, 5.0, 15)
rg_dilute = [solve_md(t, crowding=False) for t in temps]
rg_crowded = [solve_md(t, crowding=True) for t in temps]

# Plotting the Melting Curve
plt.figure(figsize=(8, 5))
plt.plot(temps, rg_dilute, 'bo-', label='Dilute Buffer')
plt.plot(temps, rg_crowded, 'ro-', label='Crowded Environment')
plt.axhline(1.5, color='k', linestyle='--', label='Folded Threshold')
plt.xlabel('Temperature (Sim Units)')
plt.ylabel('Protein Size (Radius of Gyration)')
plt.title('Protein Melting Trend: Crowding Effect')
plt.legend()
plt.grid(True)
plt.show()