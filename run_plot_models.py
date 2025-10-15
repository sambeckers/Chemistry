from __future__ import division
import os
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from kb.modeling.tools import CodeIO
from kb import KoekenBak
from kb import path
import glob as gl


###############################
### Setting in- and output -###
###############################

#- Outputfolder where the figure will be saved
savedirmain = Path('/Users/sam/Documents/GitHub/Chemistry')
model1_folder = 'complete_model_Mdot1e-5_15kms'
model2_folder = 'complete_model_Mdot1e-7_5kms'

#- Inputfile for KoekenBak
inputfile1 = str(savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot1e-5_15kms.dat')

inputfile2 = str(savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot1e-7_5kms.dat')

##- Run model
def run_model(inputfile):
  model = KoekenBak.KoekenBak(inputfile)
  model.startSession()
  return model

model1 = run_model(inputfile1)
model2 = run_model(inputfile2)


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
colors = cm.get_cmap('tab20', len(molecules)).colors  # 20 distinct colors

# #- Linestyles for each model
linestyles = ['-','--',':','-.']

def plot_model(model, model_folder):
  plt.figure(dpi=300)
  for istar, star in enumerate(model.star_grid):
    folder = Path(model_folder) / 'models' / star['LAST_CHEMISTRY_MODEL']
    fracs = CodeIO.getChemistryAbundances(str(folder / 'csfrac_smooth.out'))
    radius = CodeIO.getChemistryPhysPar(str(folder / 'csphyspar_smooth.out'),'RADIUS')
    for imol, mol in enumerate(molecules):
        if istar == 0:
          plt.loglog(radius,fracs[mol],color=colors[imol],label=mol, linestyle = linestyles[istar])
        else:
          plt.loglog(radius,fracs[mol],color=colors[imol],linestyle =linestyles[istar])
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
  plt.xlabel('Radius [cm]')
  plt.ylabel('Fractional abundance')
  plt.savefig(savedirmain / 'figures' / f'Fractional_abundances_{model_folder}.pdf', bbox_inches='tight', dpi=300)
  plt.title(model_folder)
  plt.show()

plot_model(model1, model1_folder)
plot_model(model2, model2_folder)