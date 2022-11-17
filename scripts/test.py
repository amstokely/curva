import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/andy/CLionProjects/curva')
from lib import pycurva

dcd = f'/home/andy/gb1/curva/1pgb.dcd'
pdb = f'/home/andy/gb1/curva/1pgb.pdb'
calculation = pycurva.MolecularDynamicsCalculation()
calculation.init(
    dcd=dcd,
    pdb=pdb,
    firstFrame=0,
    lastFrame=8999,
    windowSize=1000,
    name=f'1pgb',
)
for nodeIndex in range(calculation.numNodes()):
    calculation.generalizedCorrelation(
        referenceIndex=nodeIndex,
        k=8,
        cutoff=20.0,
    )
    gencor = calculation.generalizedCorrelationMatrix()[nodeIndex]
    idx = gencor.argsortDescendingAbs()
    for i in idx[:10]:
        print(gencor[i], calculation.nodes()[i], nodeIndex)
    print('')
