import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

"CALCULATION_SECTION"
traj = md.load("traj.dcd", top="topology.pdb")
avg_xyz = traj.xyz.mean(axis=0, keepdims=True)
reference = md.Trajectory(avg_xyz, traj.topology)
values = md.rmsf(traj, reference)

"PLOTTING_SECTION"
np.savetxt("rmsf.dat", values, header="rmsf", fmt="%.6f")
atoms = np.arange(1, len(values) + 1)
plt.bar(atoms, values)
plt.xlabel("Atom Index")
plt.ylabel("RMSF (nm)")
plt.grid(alpha=0.3)
plt.savefig("rmsf.png", bbox_inches="tight")
