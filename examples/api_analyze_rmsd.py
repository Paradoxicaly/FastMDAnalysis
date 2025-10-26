from fastmdanalysis import FastMDAnalysis as fastmda
from fastmdanalysis.datasets import TrpCage

#traj = "src/fastmdanalysis/data/trp_cage.dcd"
#top = "src/fastmdanalysis/data/trp_cage.pdb"

fastmda = fastmda(TrpCage.traj, TrpCage.top)
result = fastmda.analyze(
    include=["rmsd"],
    options={"rmsd": {"ref": 0, "align": True}}
)

