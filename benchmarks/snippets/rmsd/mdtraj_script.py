import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

"CALCULATION_SECTION"
traj = md.load("traj.dcd", top="topology.pdb")
reference = traj[0]
values = md.rmsd(traj, reference)

"PLOTTING_SECTION"
np.savetxt("rmsd.dat", values, header="rmsd", fmt="%.6f")
frames = np.arange(len(values))
plt.plot(frames, values, marker="o")
plt.xlabel("Frame")
plt.ylabel("RMSD (nm)")
plt.grid(alpha=0.3)
plt.savefig("rmsd.png", bbox_inches="tight")
