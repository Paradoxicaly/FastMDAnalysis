import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

"CALCULATION_SECTION"
u = mda.Universe("topology.pdb", "traj.dcd")
ca = u.select_atoms("protein and name CA")
frame_indices = range(0, u.trajectory.n_frames, 10)
coords = []
for frame in frame_indices:
    u.trajectory[frame]
    coords.append(ca.positions.copy() / 10.0)
coords = np.array(coords)
u.trajectory[0]
n_frames = coords.shape[0]
distances = np.zeros((n_frames, n_frames))
for i in range(n_frames):
    for j in range(i, n_frames):
        value = rms.rmsd(coords[i], coords[j], center=True, superposition=True)
        distances[i, j] = distances[j, i] = value
labels = DBSCAN(metric="precomputed", eps=0.3, min_samples=5).fit_predict(distances)
labels = labels - labels.min() + 1
np.savetxt("dbscan_labels.dat", np.column_stack((np.arange(n_frames), labels)), fmt="%d", header="frame cluster")
np.savetxt("dbscan_distance_matrix.dat", distances, fmt="%.6f", header="RMSD distance matrix")

"PLOTTING_SECTION"
unique = np.unique(labels)
counts = [np.sum(labels == u) for u in unique]
plt.bar(unique, counts)
plt.xlabel("Cluster ID")
plt.ylabel("Frames")
plt.grid(alpha=0.3)
plt.savefig("dbscan_pop.png", bbox_inches="tight")
