from __future__ import division
import os
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from kb.modeling.tools import CodeIO
from kb import KoekenBak
from kb.tools.io import Database
from kb import path
import glob as gl

#- Outputfolder where the figure will be saved
savedirmain = Path('/Users/sam/Documents/GitHub/Chemistry')
model1_folder = 'complete_model_Mdot1e-5_15kms_v2'
# model2_folder = 'complete_model_Mdot1e-7_5kms'

# db  = Database.Database(str(savedirmain / model1_folder / 'Chemistry_models.db'))
# db.pop('model_2025-10-17h16-58-20')
# db.sync()

#- Inputfile for KoekenBak
inputfile1 = str(savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot1e-5_15kms_v2.dat')
# inputfile2 = str(savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot1e-7_5kms.dat')

##- Run model
def run_model(inputfile):
  model = KoekenBak.KoekenBak(inputfile)
  model.startSession()
  return model

model1 = run_model(inputfile1)
# model2 = run_model(inputfile2)

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

# Parent molecules
parents = ["He", "CO", "N2", "CH4", "NH3", "H2S", "HCP", "H2O", "C2H2", "HCN", 
       "CS", "SiC2", "HCl", "HF", "C2H4", "SiO", "SiS", "Mg", "Na", "Fe"]

parents_per_pane = len(parents) // 4
parent_groups = [
    parents[0:5],
    parents[5:10],
    parents[10:15],
    parents[15:20]
]

pane_colors = [
    ['black', 'red', 'green', 'blue', 'magenta'],          
    ['black', 'orangered', 'gold', 'brown', 'hotpink'],     
    ['black', 'limegreen', 'darkgreen', 'cyan', 'teal'],    
    ['black', 'darkviolet', 'deeppink', 'navy', 'crimson']  
]

# #- Linestyles for each model
linestyles = ['-','--']

def plot_parents(model, model_folder):
  fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
  axes = axes.flatten()
  
  for istar, star in enumerate(model.star_grid):
    folder = Path(model_folder) / 'models' / star['LAST_CHEMISTRY_MODEL']
    fracs = CodeIO.getChemistryAbundances(str(folder / 'csfrac_smooth.out'))
    radius = CodeIO.getChemistryPhysPar(str(folder / 'csphyspar_smooth.out'),'RADIUS')
    
    for ipane in range(4):
      ax = axes[ipane]
      for imol, mol in enumerate(parent_groups[ipane]):
        if istar == 0:
          ax.loglog(radius, fracs[mol], color=pane_colors[ipane][imol], label=mol, linestyle=linestyles[istar])
        else:
          ax.loglog(radius, fracs[mol], color=pane_colors[ipane][imol], linestyle=linestyles[istar])
      
      if istar == len(model.star_grid) - 1:  # Only add labels after all models are plotted
        # C-rich, O-rich labels
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='gray', linestyle='-', label='C-rich'))
        handles.append(Line2D([0], [0], color='gray', linestyle='--', label='O-rich'))
        ax.legend(handles=handles, loc='best', fontsize='small')
        ax.set_xlabel('Radius [cm]')
        ax.set_ylabel('Fractional abundance')
        ax.grid(True, alpha=0.3)
  
  fig.suptitle(f'Fractional abundances - {model_folder}', fontsize=14)
  plt.tight_layout()
  plt.savefig(savedirmain / 'figures' / f'fracab_{model_folder}_parents.pdf', bbox_inches='tight', dpi=300)
  plt.show()

plot_parents(model1, model1_folder)
# plot_parents(model2, model2_folder)

# Daughters
daughters_Crich = ['C2H', 'C4H', 'C6H', 'HC3N', 'HC5N', 'HC7N']
daughters_Orich = ['SiN', 'SiC', 'OH', 'CN', 'SiOH+']
colors = ['black', 'red', 'green', 'blue', 'magenta', 'orange']

def plot_daughters(model, model_folder, daughters, rich_type):
  plt.figure(dpi=300)
  for istar, star in enumerate(model.star_grid):
    folder = Path(model_folder) / 'models' / star['LAST_CHEMISTRY_MODEL']
    fracs = CodeIO.getChemistryAbundances(str(folder / 'csfrac_smooth.out'))
    radius = CodeIO.getChemistryPhysPar(str(folder / 'csphyspar_smooth.out'),'RADIUS')
    for imol, mol in enumerate(daughters):
        if istar == 0:
          plt.loglog(radius,fracs[mol],color=colors[imol],label=mol, linestyle = linestyles[istar])
        else:
          plt.loglog(radius,fracs[mol],color=colors[imol],linestyle =linestyles[istar])
  
  # C-rich, O-rich labels
  handles, labels = plt.gca().get_legend_handles_labels()
  handles.append(Line2D([0], [0], color='gray', linestyle='-', label='C-rich'))
  handles.append(Line2D([0], [0], color='gray', linestyle='--', label='O-rich'))
  plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
  
  plt.xlabel('Radius [cm]')
  plt.ylabel('Fractional abundance')
  plt.savefig(savedirmain / 'figures' / f'fracab_{model_folder}_daughters_{rich_type}.pdf', bbox_inches='tight', dpi=300)
  plt.title(model_folder)
  plt.grid(True, alpha=0.3)
  plt.show()

plot_daughters(model1, model1_folder, daughters_Crich, 'Crich')
# # plot_daughters(model2, model2_folder, daughters_Crich, 'Crich')
plot_daughters(model1, model1_folder, daughters_Orich, 'Orich')
# # plot_daughters(model2, model2_folder, daughters_Orich, 'Orich')

# mol = 'C2H2'

# fig = plt.figure()
# for istar,star in enumerate(model1.star_grid):
#   folder = os.path.join(path.cout,'models',star['LAST_CHEMISTRY_MODEL'])+'/'  
#   radius = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','RADIUS')
#   fracs = CodeIO.getChemistryAbundances(folder+'csfrac_smooth.out')
#   plt.loglog(radius,fracs[mol]*2)

molecules = ['C2H2','C2H', 'C4H', 'C6H', 'HC3N', 'HC5N', 'HC7N']
molecules = ['SiO','SiN', 'SiC', 'OH', 'CN', 'SiOH+']
fig = plt.figure()
for istar,star in enumerate(model1.star_grid):
  for mol in molecules:
    folder = os.path.join(path.cout,'models',star['LAST_CHEMISTRY_MODEL'])+'/'  
    radius = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','RADIUS')
    fracs = CodeIO.getChemistryAbundances(folder+'csfrac_smooth.out')
    plt.loglog(radius,fracs[mol]*2, label=mol)
plt.legend()
plt.ylim(1e-12)