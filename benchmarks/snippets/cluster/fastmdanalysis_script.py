from fastmdanalysis import FastMDAnalysis

"CALCULATION_SECTION"
fastmda = FastMDAnalysis("traj.dcd", "topology.pdb", frames=(0, -1, 10), atoms="protein and name CA")

"PLOTTING_SECTION"
fastmda.analyze(include=["cluster"], options={"cluster": {"methods": ["dbscan"], "eps": 0.3, "min_samples": 5}}, slides=True)
