from fastmdanalysis import FastMDAnalysis

"CALCULATION_SECTION"
fastmda = FastMDAnalysis("traj.dcd", "topology.pdb")

"PLOTTING_SECTION"
fastmda.analyze(include=["rmsd"], options={"rmsd": {"ref": 0, "align": True}}, slides=True)
