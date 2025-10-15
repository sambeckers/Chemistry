# -*- coding: utf-8 -*-

"""
A plotting environment for Chemistry output.

Author: M. Van de Sande

"""

import os
#from scipy import array
import operator
import subprocess
#from scipy.interpolate import interp1d
import numpy as np

import kb.path
from kb.plotting.objects.PlottingSession import PlottingSession
from kb.tools.io import DataIO
from kb.plotting import Plotting2
from kb.plotting import plotFunctions
from kb.modeling.objects import Star
from kb.modeling.tools import CodeIO
from kb.modeling.tools import AnalyseIO
import glob as gl
import matplotlib as mpl
import matplotlib.pyplot as p
import math
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
#from PyPDF2 import PdfFileMerger


class PlotChem(PlottingSession):

    """ 
    Class for plotting gas lines and their information.

    """    

    def __init__(self,star_name,path_chemistry='OutputClumpy',\
                    inputfilename=None,fn_add_star=1):

        """ 
        Initializing an instance of PlotGas.

        @param star_name: name of the star from Star.dat, use default only 
                            when never using any star model specific things 
        @type star_name: string

        @keyword path_chemistry: Output modeling folder in MCMax home folder

                                    (default: 'OutputClumpy')
        @type path_chemistry: string
        @keyword inputfilename: name of inputfile that is also copied to the 
                                output folder of the plots, 
                                if None nothing is copied
                            
                                (default: None)
        @type inputfilename: string

        @keyword fn_add_star: Add the star name to the requested plot filename.
                                Only relevant if fn_plt is given in a sub method.
                            
                                (default: 1)
        @type fn_add_star: bool
                        

        """

        super(PlotChem, self).__init__(star_name=star_name,\
                                        path=path_chemistry,\
                                        code='Chemistry',\
                                        inputfilename=inputfilename,\
                                        fn_add_star=fn_add_star)
        #-- Convenience path
        kb.path.cout = os.path.join(kb.path.chemistry,self.path)



    def makeStars(self,models):

        '''
        Make a Star list based on the Chemistry model ids.

        @param models: model_ids for the Chemistry db
        @type models: list(string)

        @return: the parameter sets
        @rtype: list[Star()]

        '''

        star_grid = Star.makeStars(models=models,\
                                    id_type='Chemistry',\
                                    code='Chemistry',path=self.path)
        return star_grid



    def plotFullAnalysis(self,star_grid=[],models=[],cfg='',fn_plt='',\
                        molecules=[],print_summary=1):
        '''                      
        Plot the main destruction and production reactions of a species.
        Plots the fractional decrease/increase caused by a main reaction
        at a certain radius, for all radii. 

        @keyword star_grid: List of Star() instances. If default, model ids 
                            have to be given.
            
                            (default: [])
        @type star_grid: list[Star()]
        @keyword models: The model ids, only required if star_grid is []

                            (default: [])
        @type models: list[string]
        @keyword cfg: path to the Plotting2.plotCols config file. If default,
                        the hard-coded default plotting options are used.

                        (default: '')
        @type cfg: string        
        @keyword fn_plt: A base plot filename. Includes folder. If not, a 
                            default is added

                            (default: '')
        @type fn_plt: string
        @keyword molecules: Molecules to be plotted.

                                (default: [])
        @type molecules: list[string]
        @keyword print_summary: Print the reactions corresponding to the 
                        labels in the terminal. Always written to 
                        the outputfile.
        @type print_summary: bool         
        '''

        print('***********************************')
        print('** Plotting Full Analysis Results ')
        if not star_grid and models:
            star_grid = self.makeStars(models=models)
        elif (not models and not star_grid) or (models and star_grid):
            print('** Input is undefined or doubly defined. Aborting.')
            return

        #- Read in cfg file
        cfg_dict = Plotting2.readCfg(cfg)
        if 'filename' in cfg_dict:
            fn_plt = cfg_dict.pop('filename')
        if 'molecules' in cfg_dict:
            molecules = cfg_dict.pop('molecules')
        if 'print_summary' in cfg_dict:
            print_summary = cfg_dict.pop('print_summary')


        for istar,star in enumerate(star_grid):
            if not star['LAST_CHEMISTRY_MODEL']: continue
            if not star['FULL_ANALYSIS']:
                print('** No full analysis requested for a model. Aborting.')
                return
        
            #- Read in full analysis output, get all radii
            if star['CLUMPMODE'] == 'POROSITY':
                if star['FIC'] == 0 or star['FIC'] == 1:
                    component='minor'
                    full = CodeIO.readFullAnalysis(star,component=component)
                    radii = [full[f]['radius'] for f in full.keys()]
                else:
                    component = 'major'
                    full = CodeIO.readFullAnalysis(star,component=component)
                    radii = [full[f]['radius'] for f in full.keys()]
            else:
                component = 'major'
                full = CodeIO.readFullAnalysis(star,component=component)
                radii = [full[f]['radius'] for f in full.keys()]
            
            for m in molecules:
                    print(m)
                    #- Check if output file with the main production and destruction 
                    #  reactions troughout the wind exists. If not, make it.
                    outfile = os.path.join(kb.path.cout,'models',\
                                    star['LAST_CHEMISTRY_MODEL'])+'/OutputFullAnalysis-'\
                                    +m+'-'+component+'.txt'
                    
                    pfnp = []
                    pfnd = []
                    if os.path.isfile(outfile):
                            ##- Get main destruction and formation reactions at all radii
                            dreact,preact = AnalyseIO.getMainReactions(full,m)				
                            
                            titlep = 'production-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')
                            titled = 'destruction-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')
                            ###-- Plot of PRODUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                                    else 'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional production rates
                            #pfnp.append(plotFunctions.plotProductionReactions(radii,preact,component,m,titlep,\
                                                                            #pfn,cfg))
                            plotFunctions.plotProductionReactions(radii,preact,component,m,titlep,\
                                                                            pfn,cfg)
                            ###-- Plot of DESTRUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                                    else 'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional destruction rates
                            #pfnd.append(plotFunctions.plotDestructionReactions(radii,dreact,component,m,titled,\
                                                                            #pfn,cfg))
                            plotFunctions.plotDestructionReactions(radii,dreact,component,m,titled,\
                                                                            pfn,cfg)
                            continue
                    else:
                            AnalyseIO.writeMainReactionsFullAnalysis(star,m,component)

                            ##- Get main destruction and formation reactions at all radii
                            dreact,preact = AnalyseIO.getMainReactions(full,m)				
                            titlep = 'production-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')
                            titled = 'destruction-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')

                            ###-- Plot of PRODUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                                    else 'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional production rates
                            #pfnp.append(plotFunctions.plotProductionReactions(radii,preact,component,m,titlep,\
                                                                            #pfn,cfg))
                            plotFunctions.plotProductionReactions(radii,preact,component,m,titlep,\
                                                                            pfn,cfg)
                            ###-- Plot of DESTRUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                                    else 'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional destruction rates
                            #pfnd.append(plotFunctions.plotDestructionReactions(radii,dreact,component,m,titled,\
                                                                            #pfn,cfg))
                            plotFunctions.plotDestructionReactions(radii,dreact,component,m,titled,\
                                                                            pfn,cfg)
                    
                    
                    
            print('*** Main production and destruction processes can be found at')
            print(outfile)

            if print_summary:
                    print('\n\n')
                    file = open(outfile,'r')
                    for line in file: print(line)
                    file.close()
        
            ##- Repeat process for the minor component (if calculated)
        
            if star['CLUMPMODE'] == 'AGUNDEZ' or star['CLUMPMODE'] == 'POROSITY':
                if star['FIC'] != 0 and star['FIC'] != 1:
                    #- Read in full analysis output, get all radii
                    component = 'minor'
                    full = CodeIO.readFullAnalysis(star,component=component)
                    radii = [full[f]['radius'] for f in full.keys()]

                    for m in molecules:
                        
                        pfnp = []
                        pfnd = []
                        
                        #- Check if output file with the main production and destruction 
                        #  reactions troughout the wind exists. If not, make it.
                        outfile = os.path.join(kb.path.cout,'models',\
                                star['LAST_CHEMISTRY_MODEL'])+'/OutputFullAnalysis-'\
                                +m+'-'+component+'.txt'
                        if os.path.isfile(outfile):
                    
                            ##- Get main destruction and formation reactions at all radii
                            dreact,preact = AnalyseIO.getMainReactions(full,m)				

                            ###-- Plot of PRODUCTION reactions
                            #- Initialise figure filename	
                            titlep = 'production-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')
                            titled = 'destruction-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')

                            pfn = fn_plt+'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                        else 'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)						
                            #- Plot fractional production rates
                            #pfnp.append(plotFunctions.plotProductionReactions(radii,preact,component,m,titlep,\
                                                                                    #pfn,cfg))
                            plotFunctions.plotProductionReactions(radii,preact,component,m,titlep,\
                                                                                    pfn,cfg)
                            ###-- Plot of DESTRUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                        else 'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional destruction rates
                            #pfnd.append(plotFunctions.plotDestructionReactions(radii,dreact,component,m,titled,\
                                                                                    #pfn,cfg))
                            plotFunctions.plotDestructionReactions(radii,dreact,component,m,titled,\
                                                                                    pfn,cfg)
                            continue
                        else:
                            AnalyseIO.writeMainReactionsFullAnalysis(star,m,component)
            
                            ##- Get main destruction and formation reactions at all radii
                            dreact,preact = AnalyseIO.getMainReactions(full,m)				
                            titlep = 'production-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')
                            titled = 'destruction-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')

                            ###-- Plot of PRODUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                        else 'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)						#- Plot fractional production rates
                            #- Plot fractional production rates
                            #pfnp.append(plotFunctions.plotProductionReactions(radii,preact,component,m,titlep,\
                                                                                    #pfn,cfg))
                            plotFunctions.plotProductionReactions(radii,preact,component,m,titlep,\
                                                                                    pfn,cfg)
                            ###-- Plot of DESTRUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                        else 'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional destruction rates
                            #pfnd.append(plotFunctions.plotDestructionReactions(radii,dreact,component,m,titled,\
                                                                                    #pfn,cfg))
                            plotFunctions.plotDestructionReactions(radii,dreact,component,m,titled,\
                                                                                    pfn,cfg)    
                    
                if print_summary:
                    print('\n\n')
                    file = open(outfile,'r')
                    for line in file: print(line)
                    file.close()




    def plotFullAnalysisAccuracy(self,star_grid=[],models=[],cfg='',fn_plt='',\
                        molecules=[],print_summary=1):
        '''                      
        Plot the main destruction and production reactions of a species.
        Plots the fractional decrease/increase caused by a main reaction
        at a certain radius, for all radii. 

        @keyword star_grid: List of Star() instances. If default, model ids 
                            have to be given.
            
                            (default: [])
        @type star_grid: list[Star()]
        @keyword models: The model ids, only required if star_grid is []

                            (default: [])
        @type models: list[string]
        @keyword cfg: path to the Plotting2.plotCols config file. If default,
                        the hard-coded default plotting options are used.

                        (default: '')
        @type cfg: string        
        @keyword fn_plt: A base plot filename. Includes folder. If not, a 
                            default is added

                            (default: '')
        @type fn_plt: string
        @keyword molecules: Molecules to be plotted.

                                (default: [])
        @type molecules: list[string]
        @keyword print_summary: Print the reactions corresponding to the 
                        labels in the terminal. Always written to 
                        the outputfile.
        @type print_summary: bool         
        '''

        print('***********************************')
        print('** Plotting Full Analysis Results - Accuracy')
        if not star_grid and models:
            star_grid = self.makeStars(models=models)
        elif (not models and not star_grid) or (models and star_grid):
            print('** Input is undefined or doubly defined. Aborting.')
            return

        #- Read in cfg file
        cfg_dict = Plotting2.readCfg(cfg)
        if 'filename' in cfg_dict:
            fn_plt = cfg_dict.pop('filename')
        if 'molecules' in cfg_dict:
            molecules = cfg_dict.pop('molecules')
        if 'print_summary' in cfg_dict:
            print_summary = cfg_dict.pop('print_summary')

        reacfile = CodeIO.readRatesFile(star_grid[0])
        
        print('done')

        for istar,star in enumerate(star_grid):
            if not star['LAST_CHEMISTRY_MODEL']: continue
            if not star['FULL_ANALYSIS']:
                print('** No full analysis requested for a model. Aborting.')
                return
        
            #- Read in full analysis output, get all radii
            if star['CLUMPMODE'] == 'POROSITY':
                if star['FIC'] == 0 or star['FIC'] == 1:
                    component='minor'
                    full = CodeIO.readFullAnalysis(star,component=component)
                    radii = [full[f]['radius'] for f in full.keys()]
                else:
                    component = 'major'
                    full = CodeIO.readFullAnalysis(star,component=component)
                    radii = [full[f]['radius'] for f in full.keys()]
            else:
                component = 'major'
                full = CodeIO.readFullAnalysis(star,component=component)
                radii = [full[f]['radius'] for f in full.keys()]
            
            print('read in fa')
            print(molecules)
            for m in molecules:
                    print(m)
                    #- Check if output file with the main production and destruction 
                    #  reactions troughout the wind exists. If not, make it.
                    outfile = os.path.join(kb.path.cout,'models',\
                                    star['LAST_CHEMISTRY_MODEL'])+'/OutputFullAnalysis-'\
                                    +m+'-'+component+'.txt'
                    
                    pfnp = []
                    pfnd = []
                    if os.path.isfile(outfile):
                            ##- Get main destruction and formation reactions at all radii
                            dreact,preact = AnalyseIO.getMainReactions(full,m)				
                            
                            ##- APPEND ACCURACY
                            for ir in dreact.keys():
                                for r in dreact[ir]:
                                    r.append(reacfile[r[0]]['COEFF'][6])
                            for ir in preact.keys():
                                for r in preact[ir]:
                                    r.append(reacfile[r[0]]['COEFF'][6])
                            
                            titlep = 'production-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')
                            titled = 'destruction-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')
                            ###-- Plot of PRODUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                                    else 'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional production rates
                            plotFunctions.plotProductionReactionsAccuracy(radii,preact,component,m,titlep,\
                                                                            pfn,cfg)
                            ###-- Plot of DESTRUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                                    else 'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional destruction rates
                            plotFunctions.plotDestructionReactionsAccuracy(radii,dreact,component,m,titled,\
                                                                            pfn,cfg)
                            continue
                    else:
                            AnalyseIO.writeMainReactionsFullAnalysis(star,m,component)

                            ##- Get main destruction and formation reactions at all radii
                            dreact,preact = AnalyseIO.getMainReactions(full,m)				
                            
                            ##- APPEND ACCURACY
                            for ir in dreact.keys():
                                for r in dreact[ir]:
                                    r.append(reacfile[r[0]]['COEFF'][6])
                            for ir in preact.keys():
                                for r in preact[ir]:
                                    r.append(reacfile[r[0]]['COEFF'][6])


                            titlep = 'production-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')
                            titled = 'destruction-'+component+'-'+str(istar)+'-'+m+'-'+star['LAST_CHEMISTRY_MODEL'].replace('_','-')

                            ###-- Plot of PRODUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                                    else 'full_analysis_production_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional production rates
                            plotFunctions.plotProductionReactionsAccuracy(radii,preact,component,m,titlep,\
                                                                            pfn,cfg)
                            ###-- Plot of DESTRUCTION reactions
                            #- Initialise figure filename				
                            pfn = fn_plt+'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']  if fn_plt \
                                                    else 'full_analysis_destruction_'+component+'_'+str(istar)+'_'+m+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            #- Plot fractional destruction rates
                            plotFunctions.plotDestructionReactionsAccuracy(radii,dreact,component,m,titled,\
                                                                                pfn,cfg)
                    
            print('*** Main production and destruction processes can be found at')
            print(outfile)

            if print_summary:
                    print('\n\n')
                    file = open(outfile,'r')
                    for line in file: print(line)
                    file.close()








    def plotDestructionRates(self,star_grid=[],models=[],cfg='',\
                            fn_plt='',molecules=[],per_molecule=0,combine=0):
        '''
        Plot the production rate of a molecule vs radius.
        @keyword star_grid: List of Star() instances. If default, model ids 
                            have to be given.
                    
                            (default: [])
        @type star_grid: list[Star()]
        @keyword models: The model ids, only required if star_grid is []

                            (default: [])
        @type models: list[string]
        @keyword cfg: path to the Plotting2.plotCols config file. If default,
                        the hard-coded default plotting options are used.
            
                        (default: '')
        @type cfg: string        
        @keyword fn_plt: A base plot filename. Includes folder. If not, a 
                            default is added
            
                            (default: '')
        @type fn_plt: string
        @keyword molecules: Molecules to be plotted.

                                (default: [])
        @type molecules: list[string]
        @keyword per_molecule: Plot one molecule for all models in one figure.

                                (default: 0)
        @type per_molecule: bool
        '''
        print('***********************************')
        print('** Plotting Destruction Rates')
        if not star_grid and models:
            star_grid = self.makeStars(models=models)
        elif (not models and not star_grid) or (models and star_grid):
            print('** Input is undefined or doubly defined. Aborting.')
            return

        pfns = []
        cfg_dict = Plotting2.readCfg(cfg)
        if 'filename' in cfg_dict:
            fn_plt = cfg_dict.pop('filename')
        if 'molecules' in cfg_dict:
            molecules = cfg_dict.pop('molecules')
        if 'per_molecule' in cfg_dict:
            print_summary = cfg_dict.pop('per_molecule')
        if 'per_model' in cfg_dict:
            print_summary = cfg_dict.pop('per_model')

        #-- Some general plot settings
        extra_pars = dict()
        extra_pars['ylogscale'] = 1 
        extra_pars['xlogscale'] = 1
        extra_pars['figsize'] = (12.5,8.5)
        extra_pars['xaxis'] = 'cm'     
        yaxis = 'Destruction rate'	

        #-- Dict to keep track of all data
        ddata_major = dict()
        ddata_minor = dict()
        for istar,star in enumerate(star_grid):
            if not star['LAST_CHEMISTRY_MODEL']: continue
            if not star['FULL_ANALYSIS']:
                print('** No full analysis requested for a model. Aborting.')
                return

            ddata_major[istar] = dict()
            ddata_major[istar]['id'] = star['LAST_CHEMISTRY_MODEL']

            #- Read in full analysis output, get all radii
            component='major'
            full = CodeIO.readFullAnalysis(star,component)
            radii = [full[f]['radius'] for f in full.keys()]
            
            if star['CLUMPMODE'] == 'AGUNDEZ':
                ddata_minor[istar] = dict()
                ddata_minor[istar]['id'] = star['LAST_CHEMISTRY_MODEL']
                full_minor = CodeIO.readFullAnalysis(star,'minor')

            for m in molecules:
                #- Destruction rate at every radius
                prates = [full[d][m]['DRATE'] for d in full.keys()]
                for i,p in enumerate(prates):
                    if 'E' not in p:
                        prates[i] = '0'
                
                ddata_major[istar][m] = prates
                if star['CLUMPMODE'] == 'AGUNDEZ':
                    prates = [full_minor[d][m]['DRATE'] for d in full.keys()]
                    for i,p in enumerate(prates):
                        if 'E' not in p:
                            prates[i] = '0'
                    
                    ddata_minor[istar][m] = prates
                                
            if not per_molecule:
                #-- Collect all data
                prods_major = [ddata_major[istar][m] for m in molecules]
                ids = ddata_major[istar]['id']        		
                keytags = molecules
                
                #-- Set filename
                pfn = fn_plt if fn_plt else  m+'_'+'destruction_rates_'+component+'_'+star['LAST_CHEMISTRY_MODEL']
                pfn = self.setFnPlt(pfn)
                
                if not combine:
                    pfns.append(Plotting2.plotCols(x=radii,y=prods_major,cfg=cfg_dict,\
                                                    filename=pfn,keytags=keytags,\
                                                    plot_title=ids.replace('_','\_')+' major',\
                                                    yaxis=yaxis,**extra_pars))
                
                if star['CLUMPMODE'] == 'AGUNDEZ':
                    #-- Collect all data
                    prods_minor = [ddata_minor[istar][m] for m in molecules]
                    ids = ddata_major[istar]['id']        		
                    
                    #-- Set filename
                    pfn = fn_plt if fn_plt else m+'_'+'destruction_rates_minor'+'_'+star['LAST_CHEMISTRY_MODEL']
                    pfn = self.setFnPlt(pfn)
                    if not combine:
                        pfns.append(Plotting2.plotCols(x=radii,y=prods_minor,cfg=cfg_dict,\
                                                        filename=pfn,keytags=keytags,\
                                                        plot_title=ids.replace('_','\_')+' minor',\
                                                        yaxis=yaxis,**extra_pars))
                if combine:		
                    pfn = fn_plt if fn_plt else 'destruction_rates'
                    pfn = self.setFnPlt(pfn)		
                        
                    radii = [radii for x in xrange(2*len(molecules))]
                    prods = []
                    prods.extend(prods_major)
                    if star['CLUMPMODE'] == 'AGUNDEZ':
                        prods.extend(prods_minor)
                    
                    keytags = []
                    for ll in range(len(prods_major)):
                        keytags.append(m + 'major')
                    if star['CLUMPMODE'] == 'AGUNDEZ':
                        for ll in range(len(prods_minor)):
                            keytags.append(m + 'minor')
                
                    colors = ['r','b','k','g','c','m','y']
                    linestyles = ['--','-']
                    line_types = []
                    for ll in range(len(prods_major)):
                        line_types.append(colors[ll] + '-')
                    if star['CLUMPMODE'] == 'AGUNDEZ':
                        for ll in range(len(prods_minor)):
                            line_types.append(colors[ll] + '--')	
                    
                    pfns.append(Plotting2.plotCols(x=radii,y=prods,cfg=cfg_dict,\
                                                        filename=pfn,keytags=keytags,\
                                                        line_types=line_types,\
                                                        plot_title=ids.replace('_','\_'),\
                                                        yaxis=yaxis,**extra_pars))
                
            if per_molecule:
                for m in molecules:
                    prods_major = [dstar[m] for istar,dstar in ddata_major.items()]
                    keytags = [dstar['id'].replace('_','\_') 
                                for istar,dstar in ddata_major.items()]
        
                    #-- Make filename
                    pfn = fn_plt if fn_plt else 'destruction_rates_major'
                    pfn = self.setFnPlt(pfn,fn_suffix=m)
                    
                    if not combine:
                        pfns.append(Plotting2.plotCols(x=radii,y=prods_major,yaxis=yaxis,\
                                                        filename=pfn,keytags=keytags,\
                                                        cfg=cfg_dict,**extra_pars))  
                    
                    if star['CLUMPMODE'] == 'AGUNDEZ':
                        prods_minor = [dstar[m] for istar,dstar in ddata_minor.items()]
                        keytags = [dstar['id'].replace('_','\_') 
                                    for istar,dstar in ddata_major.items()]

                        #-- Make filename
                        pfn = fn_plt if fn_plt else 'destruction_rates_minor'
                        pfn = self.setFnPlt(pfn,fn_suffix=m)
                        if not combine:
                            pfns.append(Plotting2.plotCols(x=radii,y=prods_minor,yaxis=yaxis,\
                                                                filename=pfn,keytags=keytags,\
                                                                cfg=cfg_dict,**extra_pars)) 
                    if combine:	
                        pfn = fn_plt if fn_plt else 'destruction_rates'
                        pfn = self.setFnPlt(pfn,fn_suffix=m)	
                        prods = []
                        prods.extend(prods_major)
                        prods.extend(prods_minor)
                        
                        keys = [dstar['id'].replace('_','\_') 
                                    for istar,dstar in ddata_major.items()]
                        keytags = []
                        for ll in range(len(prods_major)):
                            keytags.append(keys[ll] + ' major')
                        if star['CLUMPMODE'] == 'AGUNDEZ':
                            for ll in range(len(prods_minor)):
                                keytags.append(keys[ll] + ' minor')
                    
                        line_types = ['r-','r--']
                        
                        pfns.append(Plotting2.plotCols(x=[radii for x in xrange(2)],\
                                                        y=prods,yaxis=yaxis,\
                                                        line_types=line_types,\
                                                        filename=pfn,keytags=keytags,\
                                                        cfg=cfg_dict,**extra_pars))
        
        print('** Plots can be found at:')
        print('\n'.join(pfns))
        print('***********************************')



    def plotProductionRates(self,star_grid=[],models=[],cfg='',\
                            fn_plt='',molecules=[],per_molecule=0,combine=0):
        '''
        Plot the production rate of a molecule vs radius.
        @keyword star_grid: List of Star() instances. If default, model ids 
                            have to be given.
                    
                            (default: [])
        @type star_grid: list[Star()]
        @keyword models: The model ids, only required if star_grid is []

                            (default: [])
        @type models: list[string]
        @keyword cfg: path to the Plotting2.plotCols config file. If default,
                        the hard-coded default plotting options are used.
            
                        (default: '')
        @type cfg: string        
        @keyword fn_plt: A base plot filename. Includes folder. If not, a 
                            default is added
            
                            (default: '')
        @type fn_plt: string
        @keyword molecules: Molecules to be plotted.

                                (default: [])
        @type molecules: list[string]
        @keyword per_molecule: Plot one molecule for all models in one figure.

                                (default: 0)
        @type per_molecule: bool
        '''
        print('***********************************')
        print('** Plotting Production Rates')
        if not star_grid and models:
            star_grid = self.makeStars(models=models)
        elif (not models and not star_grid) or (models and star_grid):
            print('** Input is undefined or doubly defined. Aborting.')
            return

        pfns = []
        cfg_dict = Plotting2.readCfg(cfg)
        if 'filename' in cfg_dict:
            fn_plt = cfg_dict.pop('filename')
        if 'molecules' in cfg_dict:
            molecules = cfg_dict.pop('molecules')
        if 'per_molecule' in cfg_dict:
            print_summary = cfg_dict.pop('per_molecule')
        if 'per_model' in cfg_dict:
            print_summary = cfg_dict.pop('per_model')

        #-- Some general plot settings
        extra_pars = dict()
        extra_pars['ylogscale'] = 1 
        extra_pars['xlogscale'] = 1
        extra_pars['figsize'] = (12.5,8.5)
        extra_pars['xaxis'] = 'cm'     
        yaxis = 'Production rate'	

        #-- Dict to keep track of all data
        ddata_major = dict()
        ddata_minor = dict()
        for istar,star in enumerate(star_grid):
            if not star['LAST_CHEMISTRY_MODEL']: continue
            if not star['FULL_ANALYSIS']:
                print('** No full analysis requested for a model. Aborting.')
                return

            ddata_major[istar] = dict()
            ddata_major[istar]['id'] = star['LAST_CHEMISTRY_MODEL']

            #- Read in full analysis output, get all radii
            component='major'
            full = CodeIO.readFullAnalysis(star,component)
            radii = [full[f]['radius'] for f in full.keys()]
            
            if star['CLUMPMODE'] == 'AGUNDEZ':
                ddata_minor[istar] = dict()
                ddata_minor[istar]['id'] = star['LAST_CHEMISTRY_MODEL']
                full_minor = CodeIO.readFullAnalysis(star,'minor')

            for m in molecules:
                #- Destruction rate at every radius
                prates = [full[d][m]['PRATE'] for d in full.keys()]
                ddata_major[istar][m] = prates
                if star['CLUMPMODE'] == 'AGUNDEZ':
                    prates = [full_minor[d][m]['PRATE'] for d in full.keys()]
                    ddata_minor[istar][m] = prates
                                                
            if not per_molecule:
                #-- Collect all data
                prods_major = [ddata_major[istar][m] for m in molecules]
                ids = ddata_major[istar]['id']        		
                keytags = molecules
                
                #-- Set filename
                pfn = fn_plt if fn_plt else m+'_'+'production_rates_'+component+'_'+star['LAST_CHEMISTRY_MODEL']
                pfn = self.setFnPlt(pfn)
                
                if not combine:
                    pfns.append(Plotting2.plotCols(x=radii,y=prods_major,cfg=cfg_dict,\
                                                    filename=pfn,keytags=keytags,\
                                                    plot_title=ids.replace('_','\_')+' major',\
                                                    yaxis=yaxis,**extra_pars))
                if star['CLUMPMODE'] == 'AGUNDEZ':
                    #-- Collect all data
                    prods_minor = [ddata_minor[istar][m] for m in molecules]
                    ids = ddata_major[istar]['id']        		
                    
                    #-- Set filename
                    pfn = fn_plt if fn_plt else m+'_'+'production_rates_minor'+'_'+star['LAST_CHEMISTRY_MODEL']
                    pfn = self.setFnPlt(pfn)
                    if not combine:
                        pfns.append(Plotting2.plotCols(x=radii,y=prods_minor,cfg=cfg_dict,\
                                                        filename=pfn,keytags=keytags,\
                                                        plot_title=ids.replace('_','\_')+' minor',\
                                                        yaxis=yaxis,**extra_pars))
                if combine:		
                    pfn = fn_plt if fn_plt else star['LAST_CHEMISTRY_MODEL']+'_production_rates'
                    pfn = self.setFnPlt(pfn)		
                        
                    radii = [radii for x in xrange(2*len(molecules))]
                    prods = []
                    prods.extend(prods_major)
                    if star['CLUMPMODE'] == 'AGUNDEZ':
                        prods.extend(prods_minor)
                    
                    keytags = []
                    for ll in range(len(prods_major)):
                        keytags.append(m + 'major')
                    if star['CLUMPMODE'] == 'AGUNDEZ':
                        for ll in range(len(prods_minor)):
                            keytags.append(m + 'minor')
                    
                    colors = ['r','b','k','g','c','m','y']
                    linestyles = ['--','-']
                    line_types = []
                    for ll in range(len(prods_major)):
                        line_types.append(colors[ll] + '-')
                    if star['CLUMPMODE'] == 'AGUNDEZ':
                        for ll in range(len(prods_minor)):
                            line_types.append(colors[ll] + '--')	
                
                    pfns.append(Plotting2.plotCols(x=radii,y=prods,cfg=cfg_dict,\
                                                        filename=pfn,keytags=keytags,\
                                                        line_types=line_types,\
                                                        plot_title=ids.replace('_','\_'),\
                                                        yaxis=yaxis,**extra_pars))
                
            if per_molecule:
                for m in molecules:
                    prods_major = [dstar[m] for istar,dstar in ddata_major.items()]
                    keytags = [dstar['id'].replace('_','\_') 
                                for istar,dstar in ddata_major.items()]
        
                    #-- Make filename
                    pfn = fn_plt if fn_plt else 'production_rates_major'
                    pfn = self.setFnPlt(pfn,fn_suffix=m)
                    
                    if not combine:
                        pfns.append(Plotting2.plotCols(x=radii,y=prods_major,yaxis=yaxis,\
                                                        filename=pfn,keytags=keytags,\
                                                        cfg=cfg_dict,**extra_pars))  
                    
                    if star['CLUMPMODE'] == 'AGUNDEZ':
                        prods_minor = [dstar[m] for istar,dstar in ddata_minor.items()]
                        keytags = [dstar['id'].replace('_','\_') 
                                    for istar,dstar in ddata_major.items()]

                        #-- Make filename
                        pfn = fn_plt if fn_plt else 'production_rates_minor'
                        pfn = self.setFnPlt(pfn,fn_suffix=m)
                        if not combine:
                            pfns.append(Plotting2.plotCols(x=radii,y=prods_minor,yaxis=yaxis,\
                                                            filename=pfn,keytags=keytags,\
                                                                cfg=cfg_dict,**extra_pars)) 
                    if combine:	
                        pfn = fn_plt if fn_plt else star['LAST_CHEMISTRY_MODEL']+'_production_rates'
                        pfn = self.setFnPlt(pfn,fn_suffix=m)	
                        prods = []
                        prods.extend(prods_major)
                        if star['CLUMPMODE'] == 'AGUNDEZ':
                            prods.extend(prods_minor)
                        
                        keys = [dstar['id'].replace('_','\_') 
                                    for istar,dstar in ddata_major.items()]
                        keytags = []
                        for ll in range(len(prods_major)):
                            keytags.append(keys[ll] + ' major')
                        if star['CLUMPMODE'] == 'AGUNDEZ':
                            for ll in range(len(prods_minor)):
                                keytags.append(keys[ll] + ' minor')
                    
                        line_types = ['r-','r--']
                        
                        pfns.append(Plotting2.plotCols(x=[radii for x in xrange(2)],\
                                                        y=prods,yaxis=yaxis,\
                                                        line_types=line_types,\
                                                        filename=pfn,keytags=keytags,\
                                                        cfg=cfg_dict,**extra_pars))
        
        print('** Plots can be found at:')
        print('\n'.join(pfns))
        print('***********************************')



    def plotNettoRates(self,star_grid=[],models=[],cfg='',\
                            fn_plt='',molecules=[],per_molecule=0,combine=0):
        '''
        Plot the production rate of a molecule vs radius.
        @keyword star_grid: List of Star() instances. If default, model ids 
                            have to be given.
                    
                            (default: [])
        @type star_grid: list[Star()]
        @keyword models: The model ids, only required if star_grid is []

                            (default: [])
        @type models: list[string]
        @keyword cfg: path to the Plotting2.plotCols config file. If default,
                        the hard-coded default plotting options are used.
            
                        (default: '')
        @type cfg: string        
        @keyword fn_plt: A base plot filename. Includes folder. If not, a 
                            default is added
            
                            (default: '')
        @type fn_plt: string
        @keyword molecules: Molecules to be plotted.

                                (default: [])
        @type molecules: list[string]
        @keyword per_molecule: Plot one molecule for all models in one figure.

                                (default: 0)
        @type per_molecule: bool
        '''
        print('***********************************')
        print('** Plotting Netto Rates (Production - Destruction)')
        if not star_grid and models:
            star_grid = self.makeStars(models=models)
        elif (not models and not star_grid) or (models and star_grid):
            print('** Input is undefined or doubly defined. Aborting.')
            return

        pfns = []
        cfg_dict = Plotting2.readCfg(cfg)
        if 'filename' in cfg_dict:
            fn_plt = cfg_dict.pop('filename')
        if 'molecules' in cfg_dict:
            molecules = cfg_dict.pop('molecules')
        if 'per_molecule' in cfg_dict:
            print_summary = cfg_dict.pop('per_molecule')
        if 'per_model' in cfg_dict:
            print_summary = cfg_dict.pop('per_model')

        #-- Some general plot settings
        extra_pars = dict()
        extra_pars['ylogscale'] = 0
        extra_pars['xlogscale'] = 1
        extra_pars['figsize'] = (12.5,8.5)
        extra_pars['xaxis'] = 'cm'     
        yaxis = 'Netto production rate'	

        #-- Dict to keep track of all data
        ddata_major_prod = dict()
        ddata_minor_prod = dict()
        ddata_major_dest = dict()
        ddata_minor_dest = dict()

        for istar,star in enumerate(star_grid):
            if not star['LAST_CHEMISTRY_MODEL']: continue
            if not star['FULL_ANALYSIS']:
                print('** No full analysis requested for a model. Aborting.')
                return

            ddata_major_prod[istar] = dict()
            ddata_major_prod[istar]['id'] = star['LAST_CHEMISTRY_MODEL']
            ddata_major_dest[istar] = dict()
            ddata_major_dest[istar]['id'] = star['LAST_CHEMISTRY_MODEL']

            #- Read in full analysis output, get all radii
            if star['CLUMPMODE'] == 'POROSITY':
                if star['FIC'] == 0 or star['FIC'] == 1:
                    component='minor'
                    full = CodeIO.readFullAnalysis(star,component=component)
                    radii = [full[f]['radius'] for f in full.keys()]
                    
                else:
                    component = 'major'
                    full = CodeIO.readFullAnalysis(star,component=component)
                    radii = [full[f]['radius'] for f in full.keys()]
                    
            else:
                component = 'major'
                full = CodeIO.readFullAnalysis(star,component=component)
                radii = [full[f]['radius'] for f in full.keys()]
                        

            if star['CLUMPMODE'] == 'AGUNDEZ' or star['CLUMPMODE'] == 'POROSITY':
                #if star['FIC'] != 0 and star['FIC'] != 1:
                                ddata_minor_prod[istar] = dict()
                                ddata_minor_dest[istar] = dict()
                                ddata_minor_prod[istar]['id'] = star['LAST_CHEMISTRY_MODEL']
                                full_minor = CodeIO.readFullAnalysis(star,'minor')
                                radii = [full[f]['radius'] for f in full_minor.keys()]

            for m in molecules:
                #- Destruction and production rate at every radius
                prates = [full[d][m]['PRATE'] for d in full.keys()]
                drates = [full[d][m]['DRATE'] for d in full.keys()]
                for i,p in enumerate(prates):
                    if 'E' not in p:
                        prates[i] = '0'
                for i,p in enumerate(drates):
                    if 'E' not in p:
                        drates[i] = '0'
                        
                ddata_major_prod[istar][m] = prates
                ddata_major_dest[istar][m] = drates
                
                if star['CLUMPMODE'] == 'AGUNDEZ' or star['CLUMPMODE'] == 'POROSITY':
                                    #if star['FIC'] != 0 and star['FIC'] != 1:
                                    prates = [full_minor[d][m]['PRATE'] for d in full_minor.keys()]
                                    drates = [full_minor[d][m]['DRATE'] for d in full_minor.keys()]
                                    ddata_minor_prod[istar][m] = prates
                                    ddata_minor_dest[istar][m] = drates
                                        
            if not per_molecule:
                            #-- Collect all data
                            prods_major = [ddata_major_prod[istar][m] for m in molecules]
                            dests_major = [ddata_major_dest[istar][m] for m in molecules]
                            ids = ddata_major_prod[istar]['id']        		
                            keytags = molecules
                            
                            netto_major = []
                            for m in molecules:
                                    netto_major.append([float(ddata_major_prod[istar][m][i]) - \
                                                                float(ddata_major_dest[istar][m][i])\
                                                                for i in range(len(ddata_major_prod[istar][m]))])

                            #-- Set filename
                            if star['CLUMPMODE'] == 'SMOOTH':
                                pfn = fn_plt if fn_plt else m+'_'+'netto_rates_'+component+'_smooth_'+star['LAST_CHEMISTRY_MODEL']
                            else:
                                pfn = fn_plt if fn_plt else m+'_'+'netto_rates_'+component+'_'+star['LAST_CHEMISTRY_MODEL']
                            pfn = self.setFnPlt(pfn)
                            
                            if not combine:
                                    pfns.append(Plotting2.plotCols(x=radii,y=netto_major,cfg=cfg_dict,\
                                                    filename=pfn,keytags=keytags,\
                                                    plot_title=ids.replace('_','\_')+' major',\
                                                    yaxis=yaxis,**extra_pars))
                                    
                            if star['CLUMPMODE'] == 'AGUNDEZ' or star['CLUMPMODE'] == 'POROSITY':
                                    #if star['FIC'] != 0 and star['FIC'] != 1:
                                            #-- Collect all data
                                    prods_minor = [ddata_minor_prod[istar][m] for m in molecules]
                                    dests_minor = [ddata_minor_dest[istar][m] for m in molecules]
                                    ids = ddata_major_prod[istar]['id']        		
                    
                                    netto_minor = []
                                    for m in molecules:
                                            netto_minor.append([float(ddata_minor_prod[istar][m][i]) - \
                                                                        float(ddata_minor_dest[istar][m][i]) \
                                                                        for i in range(len(ddata_minor_prod[istar][m]))])
        
                                    #-- Set filename
                                    #if star['CLUMPMODE'] == 'SMOOTH':
                                        #pfn = fn_plt if fn_plt else m+'_'+'netto_rates_'+component+'_smooth_'+star['LAST_CHEMISTRY_MODEL']
                                    #else:
                                        #pfn = fn_plt if fn_plt else m+'_'+'netto_rates_'+component+'_'+star['LAST_CHEMISTRY_MODEL']
                                    #pfn = self.setFnPlt(pfn)
                                    if not combine:
                                            pfns.append(Plotting2.plotCols(x=radii,y=netto_minor,cfg=cfg_dict,\
                                                                                filename=pfn,keytags=keytags,\
                                                                                plot_title=ids.replace('_','\_')+' minor',\
                                                                                yaxis=yaxis,**extra_pars))
                            if combine:		
                                    pfn = fn_plt if fn_plt else 'netto_rates'
                                    pfn = self.setFnPlt(pfn)		
                            
                                    radii = [radii for x in xrange(2*len(molecules))]
                                    netto = []
                                    netto.extend(netto_major)
                                    netto.extend(netto_minor)
                    
                                    keytags = []
                                    for ll in range(len(netto_major)):
                                            keytags.append(m + 'major')
                                    if star['CLUMPMODE'] == 'AGUNDEZ' or star['CLUMPMODE'] == 'POROSITY':
                                            if star['FIC'] != 0 and star['FIC'] != 1:
                                                    for ll in range(len(netto_minor)):
                                                            keytags.append(m + 'minor')
                    
                                    colors = ['r','b','k','g','c','m','y']
                                    linestyles = ['--','-']
                                    line_types = []
                                    for ll in range(len(netto_major)):
                                            line_types.append(colors[ll] + '-')
                                    if star['CLUMPMODE'] == 'AGUNDEZ' or star['CLUMPMODE'] == 'POROSITY':
                                            if star['FIC'] != 0 and star['FIC'] != 1:
                                                    for ll in range(len(netto_minor)):
                                                            line_types.append(colors[ll] + '--')	
                    
                                    pfns.append(Plotting2.plotCols(x=radii,y=netto,cfg=cfg_dict,\
                                                            filename=pfn,keytags=keytags,\
                                                            line_types=line_types,\
                                                            plot_title=ids.replace('_','\_'),\
                                                            yaxis=yaxis,**extra_pars))
            
            if per_molecule:
                            for m in molecules:
                                #all_colors = mpl.cm.Paired(np.linspace(0, 1, len(star_grid)))
                                prods_major = [dstar[m] for istar,dstar in ddata_major_prod.items()]
                                dests_major = [dstar[m] for istar,dstar in ddata_major_dest.items()]
                                keytags = [dstar['id'].replace('_','\_') \
                                                    for istar,dstar in ddata_major_prod.items()]

                                #-- Make filename
                                if star['CLUMPMODE'] == 'SMOOTH':
                                    pfn = fn_plt if fn_plt else 'netto_rates_major_smooth'
                                else:
                                    pfn = fn_plt if fn_plt else 'netto_rates_major'
                                pfn = self.setFnPlt(pfn,fn_suffix=m)
                                
                                netto_major = []
                                for i in range(len(dests_major)):
                                        netto_major.append([float(prods_major[i][j]) - \
                                                            float(dests_major[i][j]) 
                                                            for j in range(len(dests_major[i]))])
                
                                if not combine:
                                        pfns.append(Plotting2.plotCols(x=radii,y=netto_major,yaxis=yaxis,\
                                                                                                    filename=pfn,keytags=keytags,\
                                                                                                    cfg=cfg_dict,**extra_pars))  
                                
                                if star['CLUMPMODE'] == 'AGUNDEZ' or star['CLUMPMODE'] == 'POROSITY':
                                    if star['FIC'] != 0 and star['FIC'] != 1:
                                        prods_minor = [dstar[m] for istar,dstar in ddata_minor_prod.items()]
                                        dests_minor = [dstar[m] for istar,dstar in ddata_minor_dest.items()]
                                        keytags = [dstar['id'].replace('_','\_') 
                                                            for istar,dstar in ddata_major_prod.items()]
                                
                                        netto_minor = []
                                        for i in range(len(dests_minor)):
                                                netto_minor.append([float(prods_minor[i][j]) - \
                                                                    float(dests_minor[i][j]) 
                                                                    for j in range(len(dests_minor[i]))])

                                        #-- Make filename
                                        pfn = fn_plt if fn_plt else 'netto_rates_minor'
                                        pfn = self.setFnPlt(pfn,fn_suffix=m)
                                        if not combine:
                                            pfns.append(Plotting2.plotCols(x=radii,y=netto_minor,yaxis=yaxis,\
                                                            filename=pfn,keytags=keytags,plot_title=m,\
                                                            cfg=cfg_dict,**extra_pars)) 
                                if combine:	
                                        pfn = fn_plt if fn_plt else star['LAST_CHEMISTRY_MODEL']+'_netto_rates'
                                        pfn = self.setFnPlt(pfn,fn_suffix=m)	
                                        netto = []
                                        netto.extend(netto_major)
                                        netto.extend(netto_minor)
                        
                                        keys = [dstar['id'].replace('_','\_') 
                                                            for istar,dstar in ddata_major_prod.items()]
                                        keytags = []
                                        keytags.append(keys[0] + ' major')
                                        if star['CLUMPMODE'] == 'AGUNDEZ' or star['CLUMPMODE'] == 'POROSITY':
                                                if star['FIC'] != 0 and star['FIC'] != 1:
                                                        keytags.append(keys[0] + ' minor')
                
                                        line_types = ['r-']
                                        if star['CLUMPMODE'] == 'AGUNDEZ' or star['CLUMPMODE'] == 'POROSITY':
                                                if star['FIC'] != 0 and star['FIC'] != 1:
                                                        line_types.append('r--')							
                        
                                        pfns.append(Plotting2.plotCols(x=radii,\
                                                        y=netto,yaxis=yaxis,\
                                                        line_types=line_types,\
                                                        filename=pfn,keytags=keytags,plot_title=m,\
                                                        cfg=cfg_dict,**extra_pars))
        print('** Plots can be found at:')
        print('\n'.join(pfns))
        print('***********************************')


