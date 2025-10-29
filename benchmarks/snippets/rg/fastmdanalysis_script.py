from fastmdanalysis import FastMDAnalysis

"CALCULATION_SECTION"
fastmda = FastMDAnalysis("traj.dcd", "topology.pdb")

"PLOTTING_SECTION"
fastmda.analyze(include=["rg"], options={"rg": {"by_chain": False}}, slides=True)
