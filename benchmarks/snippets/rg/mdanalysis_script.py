import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

"CALCULATION_SECTION"
u = mda.Universe("topology.pdb", "traj.dcd")
atoms = u.atoms
masses = atoms.masses.astype(float)
total_mass = masses.sum()
values = []
for ts in u.trajectory:
    coords = atoms.positions / 10.0
    com = np.average(coords, axis=0, weights=masses)
    diff = coords - com
    squared = (diff ** 2).sum(axis=1)
    values.append(np.sqrt((squared * masses).sum() / total_mass))
values = np.array(values)

"PLOTTING_SECTION"
np.savetxt("rg.dat", values, header="rg", fmt="%.6f")
frames = np.arange(len(values))
plt.plot(frames, values, marker="o")
plt.xlabel("Frame")
plt.ylabel("Radius of Gyration (nm)")
plt.grid(alpha=0.3)
plt.savefig("rg.png", bbox_inches="tight")
