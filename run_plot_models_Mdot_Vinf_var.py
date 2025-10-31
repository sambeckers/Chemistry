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
model_folder = 'complete_model_Mdot_Vinf_Crich'
# model_folder = 'complete_model_Mdot_Vinf_Orich'

# db  = Database.Database(str(savedirmain / model_folder / 'Chemistry_models.db'))
# db.pop('model_2025-10-17h16-58-20')
# db.sync() 

#- Inputfile for KoekenBak
inputfile = str(savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot_Vinf_Crich.dat')
# inputfile = str(savedirmain / 'KoekenBak/input/' / '20251015_Sam_Mdot_Vinf_Orich.dat')

##- Run model
def run_model(inputfile):
  model = KoekenBak.KoekenBak(inputfile)
  model.startSession()
  return model

model = run_model(inputfile)

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
daughters = ['C2H', 'C4H', 'C6H', 'HC3N', 'HC5N', 'HC7N'] # Crich 
daughters = ["SiN", "SiC", "OH", "CN", "SiOH+"] # Orich

MLOSS = ["$\dot{M}=10^{-5} M_{\odot} \\rm{yr}^{-1}$", 
         "$\dot{M}=10^{-6} M_{\odot} \\rm{yr}^{-1}$", 
         "$\dot{M}=10^{-7} M_{\odot} \\rm{yr}^{-1}$"]
VELOCITY = ["$v_{\infty}=15 \\rm{km\ s}^{-1}$", 
      "$v_{\infty}=5 \\rm{km\ s}^{-1}$", 
      "$v_{\infty}=5 \\rm{km\ s}^{-1}$"]

# Colors
tab20 = mpl.colormaps['tab20b']
_oranges = mpl.colormaps['Oranges']
colors_parents = tab20(np.linspace(0, 1, len(parents)))
colors_daughters = _oranges(np.linspace(0.3, 0.95, len(daughters)))

fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300, sharey='row', sharex=True) # Each row will share an y-axis
for col, ((star), mloss_label, vinf_label) in enumerate(zip(model.star_grid, MLOSS, VELOCITY)):
  folder = os.path.join(path.cout, 'models', star['LAST_CHEMISTRY_MODEL']) + '/'
  print(folder)
  radius = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','RADIUS')
  fracs = CodeIO.getChemistryAbundances(folder+'csfrac_smooth.out')

  # Top row: parents
  ax_top = axes[0, col]
  for imol, mol in enumerate(parents):
    # Only label on the rightmost panel to keep legends there
    label = mol if col == 2 else None
    ax_top.loglog(radius, fracs[mol], color=colors_parents[imol], linewidth=1.2, label=label)
  ax_top.set_title(f"{mloss_label}, {vinf_label}", fontsize=14)
  ax_top.grid(True, which='both', alpha=0.25)
  # ax_top.set_ylim(1e-25)

  # Bottom row: daughters 
  ax_bot = axes[1, col]
  for imol, mol in enumerate(daughters):
    label = mol if col == 2 else None
    ax_bot.loglog(radius, fracs[mol], color=colors_daughters[imol % len(colors_daughters)], linewidth=1.2, label=label)
  # No mdot/vinf titles on the bottom row
  ax_bot.set_xlabel('Radius [cm]')
  ax_bot.grid(True, which='both', alpha=0.25)
  # ax_bot.set_ylim(1e-12)
axes[0, 0].set_ylabel('Fractional abundance') # Top left
axes[1, 0].set_ylabel('Fractional abundance') # Bottom left
handles_top, labels_top = axes[0, 2].get_legend_handles_labels() # Top right
if handles_top:
  axes[0, 2].legend(handles=handles_top, labels=labels_top, loc='upper right', fontsize='small')
handles_bot, labels_bot = axes[1, 2].get_legend_handles_labels() # Bottom right
if handles_bot:
  axes[1, 2].legend(handles=handles_bot, labels=labels_bot, loc='upper right', fontsize='small')
fig.text(0.52, 1.01, 'Parents', ha='center', va='top', fontsize=14)
axes[1, 1].text(0.5, 1.01, 'Daughters', transform=axes[1, 1].transAxes, ha='center', va='bottom', fontsize=14)
fig.tight_layout()
plt.savefig(savedirmain / 'figures' / f'fracab_{model_folder}_PD.pdf', bbox_inches='tight', dpi=300)
plt.show()

# mol = 'C2H2'

# fig = plt.figure()
# for istar,star in enumerate(model1.star_grid):
#   folder = os.path.join(path.cout,'models',star['LAST_CHEMISTRY_MODEL'])+'/'  
#   radius = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','RADIUS')
#   fracs = CodeIO.getChemistryAbundances(folder+'csfrac_smooth.out')
#   plt.loglog(radius,fracs[mol]*2)

# molecules = ['C2H2','C2H', 'C4H', 'C6H', 'HC3N', 'HC5N', 'HC7N']
# molecules = ['SiO','SiN', 'SiC', 'OH', 'CN', 'SiOH+']
# fig = plt.figure()
# for istar,star in enumerate(model.star_grid):
#   for mol in molecules:
#     folder = os.path.join(path.cout,'models',star['LAST_CHEMISTRY_MODEL'])+'/'  
#     print(folder)
#     radius = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','RADIUS')
#     fracs = CodeIO.getChemistryAbundances(folder+'csfrac_smooth.out')
#     plt.loglog(radius,fracs[mol]*2, label=mol)
# plt.legend()
# plt.ylim(1e-12)