# -*- coding: utf-8 -*-

"""
Module for reading code specific input/output.

Author: M. Van de Sande

"""

import os
#from scipy.integrate import trapz
#from scipy import average, argmax
import math
import numpy as np
from numpy import array

import kb.path
from kb.tools.io import DataIO
import glob as gl


def getChemistryAbundances(filename):
    
    '''
    Reads in the Chemistry abundance output, works for both 
    fractional abundances and number densities.
    
    @param filename: The filename of the abundance output 
                     (csfrac.out or csnum.out)
    @type filename: string
    
    @return: Recursive array containing the abundance per species name.
    @rtype: recarray
    
    '''
    
    ###- Open file, read in all columns
    ##data = DataIO.readCols(filename,start_row=1)
    ###- Join them into one output array
    ##C = []
    ##for c in data[1:]:
        ##C =  np.concatenate((C,c),axis = 0)

    #- Open file, read in all lines
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    
    #- Remove first line, empty strings within lists, and empty lists
    #data = [filter(None, line.split(' ')) for line in lines[1:]]
    #data = filter(None, data)
    data = [list(filter(None, line.split(' '))) for line in lines[1:]]
    data = list(filter(None, data))
    
    #- Number of columns is constant throughout the file
    if len(set([len(d) for d in data])) == 1:
        data = DataIO.readCols(filename,start_row=1)
        
        #- Join them into one output array
        C = []
        for c in data[1:]:
            C =  np.concatenate((C,c),axis = 0)
            
        #- Number of calculations per species
        c0 = data[0]
        L = np.where(np.array(c0) == c0[0])[0][1]+1        

    
    #- If the number of columns varies througout the file (e.g. by adding 
    #  species), run a more elaborate method to read in the columns    
    else:
        #- Read in first block of 10 columns and concatenate
        limit = [i for i,d in enumerate(data) if len(d) != 11][0]
        blok = list(zip(*data[:limit]))
        C = []
        for c in blok[1:]:
            C =  np.concatenate((C,c),axis = 0)
        
        # Add the appendix (with less then 10 columns)
        app = list(zip(*data[limit:]))
        for c in app[1:]:
            C =  np.concatenate((C,c),axis = 0)
        
        #- Number of calculations per species
        c0 = blok[0]
        L = np.where(np.array(c0) == c0[0])[0][1]+1        
    
    #- Put the names of the species in an array
    names = []      
    for ii in range(len(C)):
        if ii%L == 0:
            names.append(C[ii])  
    N = len(names)
    
    #- Radii of calculation
    radius = np.array(c0[1:L-1])
    radius = radius.astype(float)
    
    #- Output array: [species,[output]]
    species = np.recarray(shape = [L-2,], dtype = list(zip(names, [float]*N)))
    for ii in range(N):
        species[names[ii]] = C[(ii*L)+1:((ii+1)*L)-1].astype(float)
    
    return species



def getChemistryPhysPar(filename, keyword):
    
    '''
    Reads in the Chemistry physical output
    
    @param filename: The filename of the abundance output 
                     (csphyspar.out)
    @type filename: string
    
    @keyword keyword: The physical parameter in question. Options are:
                      RADIUS, n(H2), TEMP. A_V, RAD. FIELD, 
                      CO K(PHOT), VELOCITY
                      
    @type keyword: string
    
    @return: Recursive array containing the abundance per species name.
    @rtype: recarray
    
    '''
    
    ##- Open file, read in all columns
    #data = DataIO.readCols(filename,start_row=1)
    
    ##- Initialise keyword
    #keyword = keyword.upper()
    
    ##- Select right columm
    #c = [i for i,s in enumerate(data) if keyword in s[0]][0]
    #par = [float(p) for p in data[c][1:]]
    
    
    f = open(filename)
    data = f.readlines()
    f.close()
    
    names = data[3].split('  ')
    names = [x for x in names if x][:-1]
    
    data_test = [d.split(' ') for d in data[4:]]  
    data_new = []
    for d in data_test:
      data_new.append([float(x) for x in d if x])
    
    c = [i for i,s in enumerate(names) if keyword in s][0]
    par = [p[c] for p in data_new]
    
    
    return par



def getChemistrySpecies(filename): 
	'''
	Reads the species included in the Chemistry code.

	@param filename: The .specs file
	@type filename: string
	@keyword parents: Give parent species as output.
					  If parents=0, all species are given.
					  (default: 1)

	@return species: (Parent) species included in the Chemistry code.
	@rtype: list

	'''
	#- Read in file
	data = DataIO.readFile(filename)
	data = [x.split() for x in data]

	#- Determine the different sections in the species file
	separators = np.where([len(i)==2 for i in data])[0]

	#- Select species included
	species = data[1:separators[0]]
	species = [species[x][1] for x in range(len(species))]

	return species
    
    
def getChemistryParentSpecies(filename): 
	'''
	Reads the parent species included in the Chemistry code.

	@param filename: The .specs file
	@type filename: string
	@keyword parents: Give parent species as output.
					  If parents=0, all species are given.
					  (default: 1)

	@return species: (Parent) species included in the Chemistry code.
	@rtype: list

	'''
	#- Read in file
	data = DataIO.readFile(filename)
	data = [x.split() for x in data]

	#- Determine the different sections in the species file
	separators = np.where([len(i)==2 for i in data])[0]

	#- Select parent species
	parents = data[separators[2]:]
	parents = [parents[x][0] for x in range(len(parents))]

	return parents
    
    
def getColumnDensities(filename):
      '''
      Reads the resulting columns densities of a Chemistry model.

      @param star: KoekenBak object
      @type star: Star()

      @return coldens: Column denisties of all species.
      @rtype: dict()

      '''
      #- Read in file
      data = DataIO.readFile(filename)
      data = [x.split() for x in data]

      #- Read in output dictionary
      coldens = dict()
      for d in data:
            if len(d) == 9:
                  if 'E' in d[:3][-1]:
                        coldens[d[:3][0]] = float(d[:3][-1])
                  else:
                        coldens[d[:3][0]] = 0.0
                  if 'E' in d[3:6][-1]:
                        coldens[d[3:6][0]] = float(d[3:6][-1])
                  else:
                        coldens[d[3:6][0]] = 0.0
                  if 'E' in d[6:9][-1]:
                        coldens[d[6:9][0]] = float(d[6:9][-1])
                  else:
                        coldens[d[6:9][0]] = 0.0
            elif len(d) == 6:
                  if 'E' in d[:3][-1]:
                        coldens[d[:3][0]] = float(d[:3][-1])
                  else:
                        coldens[d[:3][0]] = 0.0
                  if 'E' in  d[3:6][-1]:
                        coldens[d[3:6][0]] = float(d[3:6][-1])
                  else:
                         coldens[d[3:6][0]] = 0.0
            elif len(d) == 3:
                  if 'E' in d[:3][-1]:
                        coldens[d[:3][0]] = float(d[:3][-1])
                  else:
                        coldens[d[:3][0]] = 0.0
      return coldens	
	  
    
def readAnalysisFile(filename):
	'''
	Reads the output of the analyse routine performed.

	@keyword filename: Analyse routine output file
	@type filename: string

	@return analyse: Dictionary containing the analyse output for every
					 molecule, eg analyse['SiO']. The analyse radius is
					 also included (analyse['radius']).
	@rtype anaylse: dict()
	'''
	#- Read in analyse.out
	data = DataIO.readFile(filename)
	data = [x.split() for x in data]

	#- Initialise output dictionary
	analyse = dict()

	#- Determine and save routine radius
	radius = float(data[0][-2])
	index = np.where([len(i)==2 for i in data])[0]
	analyse['radius'] = radius

	#- Save analyse routine output in dictionary per molecule
	for ii in range(len(index[:-1])):
		name = data[index[ii]][1][1:]
		analyse[name] = dict()
		analyse[name]['MAIN'] = data[index[ii]+1:index[ii+1]-1]
		analyse[name]['DRATE'] = data[index[ii+1]-1][1]
		analyse[name]['PRATE'] = data[index[ii+1]-1][3]
	return analyse



def readRatesFile(star):
	'''
	Reads the rates file.
	
	@keyword star: Star object of the model performed
	@type: Star()
	
	@return rates: Dict of rates file, organised by reaction number
	@rtype: dict()
	'''
	#- Read in .rates file
	filename = os.path.join(kb.path.chemistry,star['SRC_CHEMISTRY'],'rates',star['REACTIONS_FILE'])
	data = DataIO.readFile(filename)
	data = [x.split() for x in data]
	data = [x[0].split(':') for x in data]
	
	#- Initialise and define output dictionary
	rates = dict()
	for d in data:
		rates[d[0]] = dict()
		rates[d[0]]['RT'] = d[1]
		rates[d[0]]['REACTANTS'] = d[2:4]
		rates[d[0]]['PRODUCTS'] = d[4:8]
		rates[d[0]]['TFIELDS'] = d[8]
		rates[d[0]]['COEFF'] = d[9:]
		
	return rates
	
	
def readRatesFileFromFile(filename):
	'''
	Reads the rates file.
	
	@keyword star: Star object of the model performed
	@type: Star()
	
	@return rates: Dict of rates file, organised by reaction number
	@rtype: dict()
	'''
	#- Read in .rates file
	#filename = os.path.join(kb.path.chemistry,star['SRC_CHEMISTRY'],'rates',star['REACTIONS_FILE'])
	data = DataIO.readFile(filename)
	data = [x.split() for x in data]
	data = [x[0].split(':') for x in data]
	
	#- Initialise and define output dictionary
	rates = dict()
	for d in data:
		rates[d[0]] = dict()
		rates[d[0]]['RT'] = d[1]
		rates[d[0]]['REACTANTS'] = d[2:4]
		rates[d[0]]['PRODUCTS'] = d[4:8]
		rates[d[0]]['TFIELDS'] = d[8]
		rates[d[0]]['COEFF'] = d[9:]
		
	return rates


def readSingleAnalysis(star):
	'''
    Reads the output of the analyse routine performed.

    @keyword star: Star object containing the chemistry model
    @type star: Star()

    @return analyse: Dictionary containing the analyse output for every
                    	molecule, eg analyse['SiO']. The analyse radius is
                         also included (analyse['radius']).
	@rtype anaylse: dict()
	'''
	if star['PERFORM_ROUTINE'] == 0:
		print('No single analysis routine performed.')
		return

	#- Read in analyse.out
	filename = gl.glob(os.path.join(kb.path.cout,'models',
                        star['LAST_CHEMISTRY_MODEL'])+'/analyse*')[0]
	data = DataIO.readFile(filename)
	data = [x.split() for x in data]

	#- Initialise output dictionary
	analyse = dict()

	#- Determine and save routine radius
	radius = float(data[0][-2])
	index = np.where([len(i)==2 for i in data])[0]
	analyse['radius'] = radius

	#- Save analyse routine output in dictionary per molecule
	for ii in range(len(index[:-1])):
		name = data[index[ii]][1][1:]
		analyse[name] = dict()
		analyse[name]['MAIN'] = data[index[ii]+1:index[ii+1]-1]
		analyse[name]['DRATE'] = data[index[ii+1]-1][1]
		analyse[name]['PRATE'] = data[index[ii+1]-1][3]
	return analyse



def readFullAnalysis(star,component):
	'''
	Reads the output of the analyse routine performed.

    @keyword star: Star object containing the chemistry model
    @type star: Star()

    @return analyse: Dictionary containing the analyse output for every
                         molecule, eg analyse['SiO']. The analyse radius is
                         also included (analyse['radius']).
    @rtype anaylse: dict()
    '''
	if star['FULL_ANALYSIS'] == 0:
		print('No single analysis routine performed.')
		return
    #- Read in all analyse.out files and sort them
	files = gl.glob(os.path.join(kb.path.cout,'models',
                        star['LAST_CHEMISTRY_MODEL'])+'/analyse-'+component+'*')
	ints = [int(f.split('/')[-1].split('-')[-1].split('.')[0]) for f in files]
	sorted = np.argsort(ints)
	files = list(np.array(files)[sorted])
    
	#- Save analyse routine output in dictionary per iteration
	full_analyse = dict()
	for i,f in enumerate(files):
		full_analyse[i] = readAnalysisFile(f)
    
	return full_analyse
	
	
	
def readColumnDensities(star, molecule):
      '''
      Reads the resulting columns densities of a Chemistry model.

      @param star: KoekenBak object
      @type star: Star()

      @return coldens: Column denisties of all species.
      @rtype: dict()

      '''
      #-- Get filename
      filename = os.path.join(kb.path.cout,'models',
                                    star['LAST_CHEMISTRY_MODEL'])+'/cscoldens.out'
      #- Read in file
      data = DataIO.readFile(filename)
      data = [x.split() for x in data]

      #- Read in output dictionary
      coldens = dict()
      for d in data:
            if len(d) == 9:
                  if 'E' in d[:3][-1]:
                        coldens[d[:3][0]] = float(d[:3][-1])
                  else:
                        coldens[d[:3][0]] = 0.0
                  if 'E' in d[3:6][-1]:
                        coldens[d[3:6][0]] = float(d[3:6][-1])
                  else:
                        coldens[d[3:6][0]] = 0.0
                  if 'E' in d[6:9][-1]:
                        coldens[d[6:9][0]] = float(d[6:9][-1])
                  else:
                        coldens[d[6:9][0]] = 0.0
            elif len(d) == 6:
                  if 'E' in d[:3][-1]:
                        coldens[d[:3][0]] = float(d[:3][-1])
                  else:
                        coldens[d[:3][0]] = 0.0
                  if 'E' in  d[3:6][-1]:
                        coldens[d[3:6][0]] = float(d[3:6][-1])
                  else:
                         coldens[d[3:6][0]] = 0.0
            elif len(d) == 3:
                  if 'E' in d[:3][-1]:
                        coldens[d[:3][0]] = float(d[:3][-1])
                  else:
                        coldens[d[:3][0]] = 0.0
      return coldens	


def readMainProductionReactions(star,species,component):	
	"""
	Get the main production and destruction reactions for a species.
	
	@keyword star: Star object containing the model
	@type star: Star()
	@keyword species: Species of interest
	@type species: str
	@keyword component: Component of the model of interest (major or minor)
	@type component: str
	
	@return production, destruction: Reaction numbers of the main production,
									 destruction reactions
	@rtype: list(str),list(str)
	"""
	file = os.path.join(kb.path.cout,'models',star['LAST_CHEMISTRY_MODEL']+'/',\
					'OutputFullAnalysis-'+species+'-'+component+'.txt')
	
	data = DataIO.readFile(file)
	data = [d.split(' ') for d in data]
	indices = [i for i,d in enumerate(data) if d[0] == '***']

	production = []
	production.extend([d[0] for d in data[1:indices[1]]])

	return production	


def readMainDestructionReactions(star,species,component):	
	"""
	Get the main production and destruction reactions for a species.
	
	@keyword star: Star object containing the model
	@type star: Star()
	@keyword species: Species of interest
	@type species: str
	@keyword component: Component of the model of interest (major or minor)
	@type component: str
	
	@return production, destruction: Reaction numbers of the main production,
									 destruction reactions
	@rtype: list(str),list(str)
	"""
	file = os.path.join(kb.path.cout,'models',star['LAST_CHEMISTRY_MODEL']+'/',\
					'OutputFullAnalysis-'+species+'-'+component+'.txt')
	
	data = DataIO.readFile(file)
	data = [d.split(' ') for d in data]
	indices = [i for i,d in enumerate(data) if d[0] == '***']

	destruction = []
	destruction.extend([d[0] for d in data[indices[1]+1:]])

	return destruction	
	
	
	
	
	
	
	
	
	
	
	
	
	
