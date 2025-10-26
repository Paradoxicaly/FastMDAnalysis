from fastmdanalysis import FastMDAnalysis as fastmda
from fastmdanalysis.datasets import TrpCage

fastmda = fastmda(TrpCage.traj, TrpCage.top)
result = fastmda.analyze()

