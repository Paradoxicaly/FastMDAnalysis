from fastmdanalysis import FastMDAnalysis

"CALCULATION_SECTION"
fastmda = FastMDAnalysis("traj.dcd", "topology.pdb", frames=(0, -1, 10), atoms="protein and name CA")
analysis = fastmda.cluster(methods="dbscan", eps=0.3, min_samples=5)

"PLOTTING_SECTION"
results = analysis.results["dbscan"]
