# -*- coding: utf-8 -*-

"""
Main code for running the forward chemistry code of UMIST.
Heavily based on ComboCode (author: R. Lombaert)

Author: M. Van de Sande

"""

import sys
import os
import subprocess
import time

import kb.path
from kb.tools.io import DataIO
from kb.tools.numerical import Gridding
from kb.managers.PlottingManager import PlottingManager as PM
from kb.modeling.objects import Star
#from kb.statistics import ChemStats
from kb.modeling.codes import Chemistry
from kb.tools.io import Database

class KoekenBak(object):

    '''
    The interface with which to run the ComboCode package.

    '''

    def __init__(self,inputfilename):

        '''
        Initializing a KoekenBak instance.

        Once this is done, you only need to run startSession(). Then all
        methods in this class will be called according to your inputfile. Only
        run separate methods of the class if you know what you are doing!

        Input is read and parsed, and the parameter objects (Star()) are set
		The plotting manager is set.

        In the python or ipython shell you can do:
        >>> import KoekenBak
        >>> cc = KoekenBak.KoekenBak('/home/mariev/KoekenBak/input/inputChemCode.dat')

        @param inputfilename: The name of the inputfile.
        @type inputfilename: string

        '''

        self.inputfilename = inputfilename
        self.readInput()
        self.setGlobalPars()
        self.setOutputFolders()
        self.setPlotManager()
        self.setVarPars()
        self.createStarGrid()
        #- Only the extra transition pars will differ across the grid, so grab
        #- the transition list from one of the Star() objects
        self.finished = False


    def startSession(self):

        '''
        Start a ComboCode session, based on the input read upon initialisation.

        The supercomputer and model managers are set and ran.

        The plot manager, statistics module, fitter modules are ran if
        requested.

        The session ends by printing some info about the Star() objects.

        Once started, the ComboCode object cannot be started again. You will
        have to re-initialize. This will change in the future.

        '''

        if not self.finished:
            self.finished = True
            self.runChemistry()
            self.runPlotManager()
            self.runStatistics()
        else:
            print("This CC session is already finished. Please, create a new one.")



    def setGlobalPars(self):

        '''
        Set the global parameters for this CC session.

        '''

        default_global = [('chemistry',1),('statistics',1),\
                          ('path_chemistry',''),\
                          ('chemstats',0),('chemstats_molecules',[]),\
                          ('star_name','model'),\
                          ('replace_db_entry',0),('single_session',0)]
        global_pars = dict([(k,self.processed_input.pop(k.upper(),v))
                            for k,v in default_global])
        self.__dict__.update(global_pars)
        self.__setStarName()
        if (not self.path_chemistry and self.chemistry):
            raise IOError('Please define PATH_CHEMISTRY in your inputfile.')
		


    def __setStarName(self):

        '''
        Set star_name for the ComboCode object as a tuple.

        Typically this is only one name for a standard modelling session, but
        can be made multiple names as well for a statistical study.

        The ComboCode object keeps track of all the data in dicts.

        '''
        
        #if self.multiplicative_grid.has_key('STAR_NAME') \
                #or self.additive_grid.has_key('STAR_NAME'):
        if 'STAR_NAME' in self.multiplicative_grid\
                or 'STAR_NAME' in self.additive_grid:
            raise IOError('STAR_NAME incorrectly defined. Use & for grids.')
        
        if isinstance(self.star_name,str):
            self.star_name = (self.star_name,)
        
        
    def setVarPars(self):

        '''
        Define the list of variable parameters in this CC session.

        '''

        self.var_pars = [k for k in list(self.multiplicative_grid.keys()) + \
                                    list(self.additive_grid.keys())
                           if k[:5] != 'T_MAX']


    def readInput(self):

        '''
        Read input for ComboCode and return list.

        The MOLECULE, TRANSITION and R_POINTS_MASS_LOSS parameter formats are
        checked for errors in this method. If erroneous, an IOError is raised.

        '''

        input_dict = DataIO.readDict(self.inputfilename,convert_floats=1,\
                                     convert_ints=1)
        #-- keywords in multi_keys require different method
        self.processed_input = dict()
        self.multiplicative_grid = dict()
        self.additive_grid = dict()
        for k,v in input_dict.items():
            #-- Fortran input is not case sensitive. Note: use the dict
            #   value v, not input_dict[k] because of this transformation.
            k = k.upper()
            #-- Determine delimiter
            try:
                if v.find('&') != -1: delimiter = '&'
                elif v.find(';') != -1: delimiter = ';'
                elif v.find(',') != -1: delimiter = ','
                elif v.find(':') != -1: delimiter = ':'
                #-- * while no ; or , or : means multiple values for ONE model
                #   Only * star for multiplicative grid makes no sense (just
                #   give the value without delimiter)
                elif v.find('*') != -1: delimiter = '&'
                else: delimiter = ' '
            except AttributeError:
                #-- v is already a float, so can't use .find on it => no grids
                #-- no need to check the rest, continue on with the next k/v pair
                self.processed_input[k] = v
                continue
            #-- Expanding '*' entries: Assumes the value is first, the count
            #   second. Can't be made flexible, because in some cases the value
            #   cannot be discerned from the count (because both are low-value
            #   integers)
            newv = delimiter.join(\
                        [len(value.split('*')) > 1
                                  and delimiter.join([value.split('*')[0]]*\
                                                      int(value.split('*')[1]))
                                  or value
                         for value in v.split(delimiter)])
            #-- Add entries to processed_input, the multiplicative grid or the
            #-- additive grid, depending on the type of delimiter.
            if delimiter == ' ':
                self.processed_input[k] = v
            elif delimiter == ',':
                newv = [float(value)
                        for value in newv.split(',')]
                newv = Gridding.makeGrid(*newv)
                self.multiplicative_grid[k] = newv
            else:
                try:
                    if delimiter == '&':
                        newv = tuple([float(value.rstrip())
                                      for value in newv.split('&')])
                        self.processed_input[k] = newv
                    elif delimiter == ';':
                        newv = [value.rstrip() == '%' and '%' or float(value.rstrip())
                                for value in newv.split(';')]
                        self.multiplicative_grid[k] = newv
                    elif delimiter == ':':
                        newv = [value.rstrip() == '%' and '%' or float(value.rstrip())
                                for value in newv.split(':')]
                        self.additive_grid[k] = newv
                except ValueError:
                    if delimiter == '&':
                        newv = tuple([value.rstrip()
                                      for value in newv.split('&')])
                        self.processed_input[k] = newv
                    elif delimiter == ';':
                        newv = [value.rstrip()
                                for value in newv.split(';')]
                        self.multiplicative_grid[k] = newv
                    elif delimiter == ':':
                        newv = [value.rstrip()
                                for value in newv.split(':')]
                        self.additive_grid[k] = newv



    def getStars(self):

        '''
        Return the list of Star() objects for this ComboCode session.

        @return: The parameter Star() objects are returned.
        @rtype: list[Star()]

        '''

        return self.star_grid



    def createStarGrid(self):

        '''
        Create a list of Star() objects based on the inputfile that has been
        parsed with kb.readInput().

        The list of Star() objects is saved in self.star_grid, and is accessed
        through kb.getStarGrid().

        '''

        base_star = Star.Star(example_star=self.processed_input,\
                              path_chemistry=self.path_chemistry)
        if self.additive_grid:
            grid_lengths = [len(v) for v in self.additive_grid.values()]
            if len(set(grid_lengths)) != 1:
                raise IOError('The explicit parameter declaration using <:> '+\
                              'has a variable amount of options (including ' +\
                              'the R_GRID_MASS_LOSS definition). Aborting...')
            else:
                additive_dicts = [dict([(key,grid[index])
                                        for key,grid in self.additive_grid.items()])
                                  for index in range(grid_lengths[0])]
                self.star_grid = [Star.Star(example_star=base_star,\
                                         path_chemistry=self.path_chemistry,\
                                         extra_input=d)
                                  for d in additive_dicts]
        else:
            self.star_grid = [base_star]
        for key,grid in self.multiplicative_grid.items():
            self.star_grid = [Star.Star(path_chemistry=self.path_chemistry,\
                                        example_star=star,\
                                        extra_input=dict([(key,value)]))
                              for star in self.star_grid
                              for value in grid]
        #if self.processed_input.has_key('LAST_CHEMISTRY_MODEL'):
        if 'LAST_CHEMISTRY_MODEL' in self.processed_input:
            del self.processed_input['LAST_CHEMISTRY_MODEL']



    def setOutputFolders(self):

        '''
        Set the output folders.

        If the folders do not already exist, they are created.

        The locations are saved in kb.path for later use, but this is generally
        only done inside a ComboCode session. Each module sets these themselves

        '''
        
        paths = ['chemistry']
        folders = [self.path_chemistry]
        names = ['cout']
        for p,f,n in zip(paths,folders,names):
            if not getattr(self,p): continue
            path = os.path.join(getattr(kb.path,p),f)
            setattr(kb.path,n,path)
            DataIO.testFolderExistence(path)
        

    def setPlotManager(self):

        '''
        Set up the plot manager(s) for each star name.

        '''

        plot_pars = dict([(k,v)
                          for k,v in self.processed_input.items()
                          if k[0:5] == 'PLOT_' or k[0:4] == 'CFG_'])
        fn_add_star = plot_pars.pop('PLOT_FN_ADD_STAR',1)
        self.plot_manager = {sn: PM(star_name=sn,\
                                    chemistry=self.chemistry,\
                                    path_chemistry=self.path_chemistry,\
                                    inputfilename=self.inputfilename,\
                                    fn_add_star=fn_add_star,\
                                    plot_pars=plot_pars)
                             for sn in self.star_name}



    def runPlotManager(self):

        '''
        Run the plotting manager.

        '''
        
        #-- First check if any plots are requested at all
        dds = [self.plot_manager[sn].chem_pars.keys()
               for sn in self.star_name
               if self.plot_manager[sn].chem_pars.keys()]
        if not dds: return
        
        #-- Continue with the plots
        print('************************************************')
        print('****** Plotting final results.')
        print('************************************************')
        for sn in self.star_name:
            #-- Make sure resolved data files are correctly included. 
            print('** Plots for %s:'%sn)
            self.plot_manager[sn].startPlotting(self.star_grid)



    def runStatistics(self):

        '''
        Run the statistics module.

        '''

        self.chemstats = dict()
    	
        for sn in self.star_name:    
            if self.statistics and self.chemistry:
                self.chemstats_molecules = list(self.chemstats_molecules)
                ss = ChemStats.ChemStats(star_name=sn,\
                                         path_code=self.path_chemistry,\
                                         star_grid=self.star_grid,\
                                         molecules = self.chemstats_molecules)
                self.chemstats[sn] = ss


    
    def runChemistry(self):
        if self.chemistry:
            print('************************************************')
            print('** Running Chemistry ')
            print('************************************************')

            chemistry_db_path = os.path.join(kb.path.cout,'Chemistry_models.db')
            self.chem_db = Database.Database(db_path=chemistry_db_path)
            
            
            ch = Chemistry.Chemistry(path_chemistry=self.path_chemistry,\
                                    replace_db_entry = self.replace_db_entry,\
                                    db = self.chem_db,\
                                    single_session=self.single_session)
            for i,star in enumerate(self.star_grid):
                print('***********************************')
                print('** Model #%i out of %i requested models.'\
                      %(i+1,len(self.star_grid)))
                print('***********************************')

                ch.doChemistry(star)
                
                if ch.model_id:
                    star['LAST_CHEMISTRY_MODEL'] = ch.model_id



if __name__ == "__main__":
    try:
        inputfilename=sys.argv[1]
    except IndexError:
        raise IOError('Please provide an inputfilename. (syntax in the ' + \
                      'command shell: python ComboCode.py ' + \
                      '/home/robinl/inputComboCode.dat)')
    c1m = ComboCode(inputfilename)
    c1m.startSession()
