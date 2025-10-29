from fastmdanalysis import FastMDAnalysis

"CALCULATION_SECTION"
fastmda = FastMDAnalysis("traj.dcd", "topology.pdb")

"PLOTTING_SECTION"
fastmda.analyze(include=["rmsf"], slides=True)
