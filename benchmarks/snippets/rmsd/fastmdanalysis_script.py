from fastmdanalysis import FastMDAnalysis

"CALCULATION_SECTION"
fastmda = FastMDAnalysis("traj.dcd", "topology.pdb")
analysis = fastmda.rmsd(reference_frame=0)

"PLOTTING_SECTION"
analysis.plot()
