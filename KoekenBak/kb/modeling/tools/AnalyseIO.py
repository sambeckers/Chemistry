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
from kb.modeling.tools import CodeIO
import glob as gl



def writeMainReactionsFullAnalysis(star,m,component,print_output=0):
	"""
	Retrieve the main production and destruction reactions for a species 
	throughout the wind. To be used independent of plotFullAnalysisOutput.
	
	@keyword star: Star object
	@type star: Star()
	@keyword species: Species of interest
	@type species: str
	@keyword print_output: Print the location of the file
	@type print_output: bool
	"""
	#- Get rates 
	rates = CodeIO.readRatesFile(star)
	
	component = 'major' 
	#- Read in full analysis output, get all radii
	full = CodeIO.readFullAnalysis(star,component=component)
	radii = [full[f]['radius'] for f in full.keys()]

	#- Get main destruction and formation reactions at all radii
	dreact,preact = getMainReactions(full,m)		
	

	#- Initialise output file
	outfile = os.path.join(kb.path.cout,'models',\
				star['LAST_CHEMISTRY_MODEL'])+'/OutputFullAnalysis-'\
				+m+'-'+component+'.txt'
	file = open(outfile, 'w')
	#- Main production reactions
	names = getMainReactionNames(preact)
	file.write('*** Main production reactions of '+m+' - minor *** \n')
	for n in names:
		line = n + ' '+(' + ').join(rates[n]['REACTANTS'])+ \
		'  -->  '+ (' + ').join(filter(None,rates[n]['PRODUCTS']))					
		file.write(line + '\n')			
	#- Main destruction reactions	
	names = getMainReactionNames(dreact)
	file.write('*** Main destruction reactions of '+m+' - minor *** \n')
	for n in names:
		line = n + ' '+\
		(' + ').join(rates[n]['REACTANTS'])+ \
		'  -->  '+ (' + ').join(filter(None,\
		rates[n]['PRODUCTS']))					
		file.write(line + '\n')	
	file.close()
	
	if print_output:
		print('** Main production and destruction reactions of '+m+'can be found at:')
		print(outfile)
			
	if star['CLUMPMODE'] == 'AGUNDEZ':
		component = 'minor'
		#- Read in full analysis output, get all radii
		full = CodeIO.readFullAnalysis(star,component=component)
		radii = [full[f]['radius'] for f in full.keys()]

		#- Get main destruction and formation reactions at all radii
		dreact,preact = getMainReactions(full,m)		
		#- Initialise output file
		outfile = os.path.join(kb.path.cout,'models',\
					star['LAST_CHEMISTRY_MODEL'])+'/OutputFullAnalysis-'\
					+m+'-'+component+'.txt'
		file = open(outfile, 'w')
		#- Main production reactions
		names = getMainReactionNames(preact)
		file.write('*** Main production reactions of '+m+' - minor *** \n')
		for n in names:
			line = n + ' '+(' + ').join(rates[n]['REACTANTS'])+ \
			'  -->  '+ (' + ').join(filter(None,rates[n]['PRODUCTS']))					
			file.write(line + '\n')			
		#- Main destruction reactions	
		names = getMainReactionNames(dreact)
		file.write('*** Main destruction reactions of '+m+' - minor *** \n')
		for n in names:
			line = n + ' '+\
			(' + ').join(rates[n]['REACTANTS'])+ \
			'  -->  '+ (' + ').join(filter(None,\
			rates[n]['PRODUCTS']))					
			file.write(line + '\n')	
		file.close()
				
			
		
		
		
		
		
def getMainReactions(full,molec):
	'''
	Extracts the main destruction and production reactions for a species
	throughout the outflow, given in the analysis output.
	
	@keyword data: Full analysis output of the model						
	@type data: dict()
	@keyword molec: Species for which you want the main destruction and 
					production reactions.
	@type molec: string
	
	@return dreact: Dict of main destruction reaction. 
					The keys are the calculation step/radius.
	@rtype dreact: dict()
	
	@return preact: Dict of main production reactions.
					The keys are the calculation step/radius.
	@rtype preact: dict()
	'''
	dreact = dict()
	preact = dict()
	
	for d in full.keys():
		dreact[d] = []
		preact[d] = []
		for react in full[d][molec]['MAIN']: 
			if react[-1] != '******':
				if float(react[-1]) < 0:
					dreact[d].append([react[0],react[-1]])
				else:
					preact[d].append([react[0],react[-1]])
	
	return dreact,preact
	
	
def getMainReactionNames(preact):
	"""
	Find the main production/destrucion reactions of a molecule

	@keyword preact: Main production/destrucion reactions of a molecule
					 (from CodeIO.getMainReactions(full analysis output, molec))
	@type preact: dict()
	
	@return names: Reaction numbers of the main production/destruction reactions
	@rtype names: list()
	"""
	
	names = []
	for pr in preact.values():
		for i in pr:
			names.append(i[0])
	names = list(set(names))
	return names
	

def getSingleProductionRate(full,m,react):
	"""
	Get the reaction rate of a single reaction.
	
	@keyword full: Output of the full analysis routine
	@type full: dict()
	@keyword m: Molecule of interest
	@type m: str
	@keyword react: Number of the reaction of interest
	@type react: str 
	
	@return rate: Production rate of the reaction throughout the outflow
				  cm-3 s-1
	@rtype rate: list()
	"""
	
	dreact,preact = CodeIO.getMainReactions(full,m)	
	prates = [full[d][m]['PRATE'] for d in full.keys()]
			
	rates = []
	for i,prod in preact.items():
		for pr in prod:		
			if pr[0] == react:
				rates.append(float(pr[1])/float(prates[i]))
		if react not in [pr[0] for pr in prod]:
			rates.append(0)	
	return rates
	
def getSingleDestructionRate(full,m,react):
	"""
	Get the reaction rate of a single reaction.
	
	@keyword full: Output of the full analysis routine
	@type full: dict()
	@keyword m: Molecule of interest
	@type m: str
	@keyword react: Number of the reaction of interest
	@type react: str 
	
	@return rate: Production rate of the reaction throughout the outflow
				  cm-3 s-1
	@rtype rate: list()
	"""
	
	dreact,preact = CodeIO.getMainReactions(full,m)	
	prates = [full[d][m]['DRATE'] for d in full.keys()]
			
	rates = []
	for i,prod in dreact.items():
		for pr in prod:		
			if pr[0] == react:
				rates.append(float(pr[1])/float(prates[i]))
		if react not in [pr[0] for pr in prod]:
			rates.append(0)	
	return rates
		




	
	
	
