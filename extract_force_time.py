#Reads the ODB (output database) files
#Extracts RF1 (Reaction Force along loading direction) data for the boundary condition
#Saves Force-Time data as a text file
from abaqus import *
from abaqusConstants import *
from odbAccess import *
from caeModules import *
import numpy as np
import visualization
path='E:/temp/asc-off/Tension-parametric-asc_ASC_parametric_c'                      #define path to the simulations ODB files

for i in range(1,1100):                                                             #set range from 1 to number of completed simulations. Each file is sequentially read and force data extracted
    odbName = path+str(i)
    odb = openOdb(odbName + '.odb')                                                 #Open ODB file in ABAQUS
    savefile='asc-para-off1-c'                                                      #Name for the text file where the force data will be written
    session.viewports['Viewport: 1'].setValues(displayedObject=odb)
    xyList = xyPlot.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=((  #Obtain Nodal Force Data along the loading direction
        'RF', NODAL, ((COMPONENT, 'RF1'), )), ), nodeSets=(
        'ASSEMBLY_CONSTRAINT-1_REFERENCE_POINT', ))                                 #ASSEMBLY_CONSTRAINT-1_REFERENCE_POINT refers to the node set name 
    #print(xyList)
    savefilename=savefile+str(i)+'.txt'                                             #save the obtained data
    session.writeXYReport(fileName=savefilename, xyData=xyList)
    odb.close()                                                                     #Close ODB file