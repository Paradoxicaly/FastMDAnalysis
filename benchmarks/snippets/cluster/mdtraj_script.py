import mdtraj as md
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

"CALCULATION_SECTION"
traj = md.load("traj.dcd", top="topology.pdb")[::10]
ca = traj.topology.select("protein and name CA")
subtraj = traj.atom_slice(ca)
n_frames = subtraj.n_frames
distances = np.zeros((n_frames, n_frames))
for i in range(n_frames):
    distances[i] = md.rmsd(subtraj, subtraj[i])
distances = 0.5 * (distances + distances.T)
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
