import numpy as np
import subprocess

'''
# 1. Define your two arrays
distances = np.linspace(50.0, 200.0, 10)  # 10 steps from 50 to 200 Mpc
thetas = np.linspace(0.0, 1.57, 10)       # 10 steps from 0 to pi/2 rad

total_runs = len(distances) * len(thetas)
current_run = 1

# 2. Loop through the grid
for d in distances:
    for t in thetas:
        print(f"--- Starting Run {current_run}/{total_runs} | dL: {d:.1f}, Theta: {t:.2f} ---")
        
        # This acts like you typing "python main_run.py --dl 50 --theta 0.0" in the terminal
        subprocess.run([
            "python", "bilby_cbc_param_estimation.py", 
            "--dl", str(d), 
            "--theta", str(t)
        ])
        
        current_run += 1

print("10x10 Grid Sweep Complete!")
'''

import subprocess
import itertools
import time

distances = np.linspace(50.0, 200.0, 10)
thetas = np.linspace(0.0, 1.57, 10)  
cases = list(itertools.product(distances, thetas))

max_concurrent = 3
running = []

for dl, theta in cases:
    while len(running) >= max_concurrent:
        running = [p for p in running if p.poll() is None]  # drop finished
        time.sleep(10)
    logfile = open(f"logs/dl{int(dl)}_th{theta:.2f}.log", "w")
    p = subprocess.Popen(
        ["python", "bilby_cbc_param_estimation.py", "--dl", str(dl), "--theta", str(theta)],
        stdout=logfile, stderr=subprocess.STDOUT
    )
    running.append(p)

for p in running:
    p.wait()
