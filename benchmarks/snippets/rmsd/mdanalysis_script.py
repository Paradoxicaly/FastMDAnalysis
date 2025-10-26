import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import matplotlib.pyplot as plt

"CALCULATION_SECTION"
universe = mda.Universe("topology.pdb", "traj.dcd")
calc = rms.RMSD(universe, ref_frame=0, select="all")
calc.run()
values = calc.results.rmsd[:, 2]

"PLOTTING_SECTION"
np.savetxt("rmsd.dat", values, header="rmsd", fmt="%.6f")
frames = np.arange(len(values))
plt.plot(frames, values, marker="o")
plt.xlabel("Frame")
plt.ylabel("RMSD (nm)")
plt.grid(alpha=0.3)
plt.savefig("rmsd.png", bbox_inches="tight")
