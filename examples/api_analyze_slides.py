from fastmdanalysis import FastMDAnalysis as fastmda
from fastmdanalysis.datasets import TrpCage

#traj = "src/fastmdanalysis/data/trp_cage.dcd"
#top = "src/fastmdanalysis/data/trp_cage.pdb"

fastmda = fastmda(TrpCage.traj, TrpCage.top)
result = fastmda.analyze(
    exclude=["dimred"],
    options={"rmsd": {"ref": 0, "align": True}},
    slides=True     # or slides="results.pptx"
)

#print(result.get("slides").value if result.get("slides") and result["slides"].ok else "No deck")
