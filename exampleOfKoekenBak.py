from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as p
from kb.modeling.tools import CodeIO
from kb import KoekenBak
from kb import path
import matplotlib as mpl
import glob as gl

###############################
### Setting in- and output -###
###############################

#- Outputfolder where the figure will be saved
savedirmain = '/Users/sam/Documents/GitHub/Chemistry'

#- Filename of the figure 
filename = 'CompRates-fracabs'

#- Inputfile for KoekenBak
inputfile = '/Users/sam/Documents/GitHub/Chemistry/KoekenBak/input/' \
'20251010_Sam_tester.dat'



###########################
###- Running the model -###
###########################

##- Run model
model = KoekenBak.KoekenBak(inputfile)
model.startSession()




#################################
###- Some KoekenBak features -###
#################################

###- These aren't useful for plotting, just to show you how it works and what it can do

#- The outputfolder you specified
path.cout

#- The folder where csmodel is located
path.csource

#- Each model is located in a Star object. You can have a look by typing
for istar,star in enumerate(model.star_grid):
    print(istar, star)
# star is a dictionary containing all your input parameters, as well as the location of the model ouput in the outputfolder you specified
star['LAST_CHEMISTRY_MODEL']



#- The folder containing the model output
folder =  os.path.join(path.cout,'models',star['LAST_CHEMISTRY_MODEL'])+'/'  

##- Easy reading of the output is done using CodeIO
#- Take a look at the files it opens!

##- Physical parameters
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



##############################
### Let's plot some things ###
##############################


###- A simple plot

p.figure()
#- Loop over different models in your inputfile (if you have a grid, this plots all models on one plot)
for istar,star in enumerate(model.star_grid): 
  #- Retrieve the output folder
  folder = os.path.join(path.cout,'models',star['LAST_CHEMISTRY_MODEL'])+'/'  
  #- Read in fractional abundances and radii
  fracs = CodeIO.getChemistryAbundances(folder+'csfrac_smooth.out')
  radius = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','RADIUS')
  #- Plot the fractional abundance of H2O
  p.loglog(radius,fracs['H2O'],linestyle ='-')
p.show()  



###- A more advanced plot

#- Molecules you want to plot
molecules = ['SiO','C2H2']

#- Colours for each molecule 
colors = p.rcParams['axes.prop_cycle'].by_key()['color']

#- Linestyles for each model
linestyles = ['-','--',':','-.']

p.figure()
for istar,star in enumerate(model.star_grid): 
  folder = os.path.join(path.cout,'models',star['LAST_CHEMISTRY_MODEL'])+'/'  
  fracs = CodeIO.getChemistryAbundances(folder+'csfrac_smooth.out')
  radius = CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','RADIUS')
  
  #- Loop over models
  for imol,mol in enumerate(molecules):
      #- Only label the first model, so that your legend isn't messy
    if istar == 0:
      p.loglog(radius,fracs[mol],color=colors[imol],label=mol,linestyle = linestyles[istar])
    else:
      p.loglog(radius,fracs[mol],color=colors[imol],linestyle =linestyles[istar])
p.legend()
p.show()





######################################
### Scripts for dust-gas chemistry ###
######################################

# Prefix 'G' = ice, prefix 'R' = refractory


###- A function to get all species, all ices and refractories
#- Remove the refr if you're not using a model that includes them!

def getSpecies(star):

  folder = os.path.join(savedirmain,star['SRC_CHEMISTRY'],'specs')+'/'  
  specs = CodeIO.getChemistrySpecies(folder + star['SPECIES_FILE'])

  refr = [s for s in specs if s[0] == 'R']
  ice = [s for s in specs if s[0] == 'G']
  
  return specs,refr,ice


###- A function to get the total ice number density 
#- You can modify this to give you a total whatever, just change the species in "ices"

def getAllIce(nums,ices,radius):
  #- Get all things sticking to the dust
  
  allice = []
  for irad,rad in enumerate(radius):
    temp = []
    for imol,mol in enumerate(ices):
      temp.append(nums[mol][irad])
    allice.append(sum(temp))
  
  return allice

#- Dividing by hnr gives you the total ice fractional abundance (relative to H2)




###- A function that calculates the total density of binding sites
#- Note that you need to give it a "star"!

def calcDensites(star):
  folder = os.path.join(path.cout,'models',star['LAST_CHEMISTRY_MODEL'])+'/'  
  hnr = np.array(CodeIO.getChemistryPhysPar(folder+'csphyspar_smooth.out','H2'))

  C = 3.*(4.+star['GRAINEXP'])*star['DTG']*mu*mh*hnr\
    /(4.*np.pi*star['RHODUST']*(np.power(star['AMAX'],4.+star['GRAINEXP']) \
      - np.power(star['AMIN'],4.+star['GRAINEXP'])))
  densites = C*1e15*4.*np.pi*(np.power(star['AMAX'],3.+star['GRAINEXP']) \
    - np.power(star['AMIN'],3.+star['GRAINEXP']))/(3.+star['GRAINEXP'])
  
  return densites


###- To calculate the number of ice monolayers:
# nmono_ice = allice/densites

















