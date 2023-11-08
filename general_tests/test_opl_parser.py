'''
Created on April 25, 2022

@author: mandd, wangc
'''

import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

cwd = os.path.dirname(__file__)
frameworkDir = os.path.abspath(os.path.join(cwd, os.pardir, 'src'))
sys.path.append(frameworkDir)

from dackar.utils.opm.OPLparser import OPMobject

pump_opl_file = os.path.abspath(os.path.join(cwd, os.pardir,'opm_models', 'pump_opl.html'))
opm = OPMobject(pump_opl_file)

'''Testing workflow '''
opm.OPLentityParser()
opm.OPLtextParser()
opm.OPLparser()
print(opm.opmGraph.nodes(data=True))
formList = opm.returnObjectList()
functionList = opm.returnProcessList()
attributeList = opm.returnAttributeList()
opmGraph = opm.returnGraph()
externalLinks = opm.returnsExternalLinks()
nx.draw_networkx(opmGraph)
ax = plt.gca()
plt.axis("off")
plt.show()
