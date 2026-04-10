import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# USER PARAMETERS (Change these to test!)
# ==========================================
BEADS = 30          # N: More beads = more realistic, but slower
STEPS = 30000       # Higher = smoother plots (less noise)
FOLD_STRENGTH = 30  # Strength of native contacts (Glues it together)
CROWDING_SIZE = 2.4 # Smaller = more crowded/stable
TEMP_MIN = 20        # Start Temp (Celsius)
TEMP_MAX = 40       # End Temp (Celsius)
# ==========================================

def run_md_simulation(temp_c, is_crowded=False):
    # Convert Celsius to Simulation Energy
    temp = (temp_c + 273.15) / 110 
    dt = 0.01
    
    # Initialize coordinates (linear chain)
    pos = np.zeros((BEADS, 3))
    pos[:, 0] = np.linspace(0, 2, BEADS)
    vel = np.random.normal(0, np.sqrt(temp), (BEADS, 3))
    
    rg_history = []
    
    for s in range(STEPS):
        forces = np.zeros((BEADS, 3))
        
        # 1. Connectivity (Harmonic Bonds)
        for i in range(BEADS - 1):
            r = pos[i+1] - pos[i]
            d = np.linalg.norm(r)
            forces[i] += 250 * (d - 1.0) * (r / d)
            forces[i+1] -= 250 * (d - 1.0) * (r / d)
            
        # 2. Native Contacts (Folding logic - every 4th bead attracts)
        for i in range(BEADS - 4):
            r = pos[i+4] - pos[i]
            d = np.linalg.norm(r)
            if d < 4.0:
                forces[i] += FOLD_STRENGTH * (d - 1.3) * (r / d)
                forces[i+4] -= FOLD_STRENGTH * (d - 1.3) * (r / d)

        # 3. Crowding Logic (Steric Constraint)
        if is_crowded:
            for i in range(BEADS):
                dist = np.linalg.norm(pos[i])
                if dist > CROWDING_SIZE:
                    forces[i] -= 60.0 * (dist - CROWDING_SIZE) * (pos[i]/dist)

        # Thermostat and Integration
        vel = (vel + forces * dt) * 0.94 
        pos += vel * dt
        
        # Only collect data after the protein "settles" (Equilibrium)
        if s > (STEPS // 2):
            rg = np.sqrt(np.mean(np.sum((pos - np.mean(pos, 0))**2, 1)))
            rg_history.append(rg)
            
    return np.mean(rg_history)

# Running the loop
temps_c = np.linspace(TEMP_MIN, TEMP_MAX, 15)
print("Simulating Dilute environment...")
rg_dilute = np.array([run_md_simulation(t, False) for t in temps_c])
print("Simulating Crowded environment...")
rg_crowded = np.array([run_md_simulation(t, True) for t in temps_c])

# Normalization (Paper Style)
def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

f_u_dilute = normalize(rg_dilute)
f_u_crowded = normalize(rg_crowded)

# Plotting Results
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(temps_c, f_u_dilute, 'ko-', label='Dilute Buffer', linewidth=2)
ax.plot(temps_c, f_u_crowded, 'ro-', label='Crowded Environment', linewidth=2)

ax.axhline(0.5, color='blue', linestyle='--', alpha=0.6, label='Tm Transition')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Fraction Unfolded (Normalized $R_g$)')
ax.set_title(f'Protein Stability Analysis ($N={BEADS}$)')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.5)

plt.show()