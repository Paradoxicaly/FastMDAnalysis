from fastmdanalysis import FastMDAnalysis

"CALCULATION_SECTION"
fastmda = FastMDAnalysis("traj.dcd", "topology.pdb")
analysis = fastmda.rg()

"PLOTTING_SECTION"
analysis.plot()
