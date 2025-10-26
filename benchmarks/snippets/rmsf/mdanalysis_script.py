import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import matplotlib.pyplot as plt

"CALCULATION_SECTION"
u = mda.Universe("topology.pdb", "traj.dcd")
analysis = rms.RMSF(u.atoms).run()
values = analysis.results.rmsf / 10.0

"PLOTTING_SECTION"
np.savetxt("rmsf.dat", values, header="rmsf", fmt="%.6f")
atoms = np.arange(1, len(values) + 1)
plt.bar(atoms, values)
plt.xlabel("Atom Index")
plt.ylabel("RMSF (nm)")
plt.grid(alpha=0.3)
plt.savefig("rmsf.png", bbox_inches="tight")
