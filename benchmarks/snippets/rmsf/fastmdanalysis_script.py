from fastmdanalysis import FastMDAnalysis

"CALCULATION_SECTION"
fastmda = FastMDAnalysis("traj.dcd", "topology.pdb")
analysis = fastmda.rmsf()

"PLOTTING_SECTION"
analysis.plot()
