from __future__ import division
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from kb.modeling.tools import CodeIO
from kb import KoekenBak
from kb import path
import matplotlib as mpl
import glob as gl

###############################
### Setting in- and output -###
###############################

#- Outputfolder where the figure will be saved
savedirmain = Path('/Users/sam/Documents/GitHub/Chemistry')

#- Filename of the figure 
filename = 'CompRates-fracabs'

#- Inputfile for KoekenBak
inputfile1 = savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot1e-5_15kms.dat'

inputfile2 = savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot1e-7_5kms.dat'

##- Run model
def run_model(inputfile):
  model = KoekenBak.KoekenBak(inputfile)
  model.startSession()
  return model

model1 = run_model(inputfile1)
model2 = run_model(inputfile2)

#- The outputfolder you specified
path.cout
#- The folder where csmodel is located
path.csource


#- The folder containing the model output
folder =  os.path.join(path.cout,'models',star['LAST_CHEMISTRY_MODEL'])+'/'  
print(folder)

def phys_params():
  radius = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','RADIUS') # radius [cm]
  hnr = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','(H2)') # H2 number density [cm-3]
  tempgas = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','TEMP') # gas temperature [K]
  tempdust = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','T_DUST') # dust temperature [K]
  #- You probably won't need these
  av = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','A_V') # visual extinction [mag]
  radfield = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','FIELD') # decrease in interstellar radiation field

  ##- Fractional abundances
  #- This returns a dictionary. Each entry contains the abundances relative to H2 of a species
  fracs = CodeIO.getChemistryAbundances(folder+'csfrac_smooth.out')

  ##- Number densities
  #- This returns a dictionary. Each entry contains the number density [cm-3]
  nums = CodeIO.getChemistryAbundances(folder+'csnum_smooth.out')
  return radius,hnr,tempgas,tempdust,av,radfield,fracs,nums




###- A more advanced plot
# #- Molecules you want to plot
molecules = molecules = [
    "He",
    "CO",
    "N2",
    "CH4",
    "NH3",
    "H2S",
    "HCP",
    "H2O",
    "C2H2",
    "HCN",
    "CS",
    "SiC2",
    "HCl",
    "HF",
    "C2H4",
    "SiO",
    "SiS",
    "Mg",
    "Na",
    "Fe"
]

# #- Colours for each molecule 
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# #- Linestyles for each model
linestyles = ['-','--',':','-.']

plt.figure(dpi=300)
for istar, star in enumerate(model1.star_grid):
  folder = Path(path.cout) / 'models' / star['LAST_CHEMISTRY_MODEL']
  fracs = CodeIO.getChemistryAbundances(folder / 'csfrac_smooth.out')
  radius = CodeIO.getChemistryPhysPar(folder / 'csphyspar_smooth.out','RADIUS')
  for imol, mol in enumerate(molecules):
      if istar == 0:
        plt.loglog(radius,fracs[mol],color=colors[imol],label=mol,linestyle = linestyles[istar])
      else:
        plt.loglog(radius,fracs[mol],color=colors[imol],linestyle =linestyles[istar])
plt.legend()
plt.show()

