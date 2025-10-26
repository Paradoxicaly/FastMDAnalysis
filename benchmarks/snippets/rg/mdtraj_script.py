import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

"CALCULATION_SECTION"
traj = md.load("traj.dcd", top="topology.pdb")
values = md.compute_rg(traj)

"PLOTTING_SECTION"
np.savetxt("rg.dat", values, header="rg", fmt="%.6f")
frames = np.arange(len(values))
plt.plot(frames, values, marker="o")
plt.xlabel("Frame")
plt.ylabel("Radius of Gyration (nm)")
plt.grid(alpha=0.3)
plt.savefig("rg.png", bbox_inches="tight")
