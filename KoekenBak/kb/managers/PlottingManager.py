# -*- coding: utf-8 -*-

"""
Interface for plotting.

Author: R. Lombaert

"""

import os

import kb.path
from kb.plotting.objects import PlotChem


class PlottingManager():
    
    """ 
    An interface for managing requested plots for a CC session.
    
    """
    
    def __init__(self,star_name,\
                 chemistry=False,\
                 path_chemistry='OutputClumpy',\
                 inputfilename='inputComboCode.dat',spire=None,fn_add_star=1,\
                 plot_pars=dict(),sed=None):
                
        """ 
        Initializing a PlottingManager instance.
        
        @keyword star_name: name of the star from Star.dat, use default only 
                            when never using any star model specific things 
                                  
                            (default: "model")
        @type star_name: string
        @keyword inputfilename: name of inputfile that is also copied to the 
                                output folder of the plots, 
                                if None nothing is copied
                                
                                (default: None)
        @type inputfilename: string
        @keyword mcmax: Running MCMax?
        
                        (default: 0)
        @type mcmax: bool
        @keyword gastronoom: Running GASTRoNOoM?
        
                             (default: 0)
        @type gastronoom: bool
        @keyword path_mcmax: modeling folder in MCMax home
        
                             (default: 'runTest')
        @type path_mcmax: string
        @keyword path_gastronoom: modeling folder in GASTRoNOoM home
        
                                  (default: 'runTest')
        @type path_gastronoom: string
        @keyword pacs: A Pacs() object for managing data and model handling for
                       PACS spectra. None if not applicable
                            
                       (default: None)
        @type pacs: Pacs()
        @keyword spire: A Spire() object for managing data and model handling
                        for SPIRE spectra. None if not applicable
                            
                        (default: None)
        @type spire: Spire()
        @keyword sed: The SED for managing data and model handling
                      for SED spectra/photometry. None if not applicable
                          
                      (default: None)
        @type sed: Sed()
        @keyword fn_add_star: Add the star name to the requested plot filename.
                              Only relevant if fn_plt is given in a sub method.
                              
                              (default: 1)
        @type fn_add_star: bool
        @keyword plot_pars: dictionary with all the plotting parameters that
                            turn on or off plotting modules. By default they
                            are all turned off.
                                  
                            (default: dict())
        @type plot_pars: dict
        
        """
        
        self.chem_pars = dict()
        self.chem_cfg = dict()
        for k,v in plot_pars.items():
            if k[0:15] == 'PLOT_CHEMISTRY_' and v:
                self.chem_pars[k.replace('_CHEMISTRY','',1)] = v
            elif k[0:14] == 'CFG_CHEMISTRY_' and v:
                self.chem_cfg[k.replace('_CHEMISTRY','',1)] = v
        self.chemistry = chemistry
        if self.chemistry:
            self.plotter_chem = PlotChem.PlotChem(star_name=star_name,\
                                                  path_chemistry=path_chemistry,\
                                                  inputfilename=inputfilename,
                                                  fn_add_star=fn_add_star)
        else:
            self.plotter_chem = None
        

    
    def startPlotting(self,star_grid,iterative=0):
        
        """ 
        Start plotting PLOT_INPUT requests for those models that are available 
        (i.e. gastronoom and/or mcmax).
        
        @param star_grid: list of stars to be plotted
        @type star_grid: list[Star()]
        
        @keyword iterative: if true the old grids are plotted on a 
                            model_iteration per model_iteration basis, only
                            works for MCMax models for now. 
                                  
                            (default: 0)
        @type iterative: int
        
        """
        
        if self.chemistry:
            for k in self.chem_pars:
                method_name = 'plot' + \
                              ''.join([w.capitalize() 
                                       for w in k.replace('PLOT_','')\
                                                 .split('_')])
                thisMethod = getattr(self.plotter_chem,method_name)
                thisMethod(star_grid=star_grid,\
                           cfg=self.chem_cfg.get(k.replace('PLOT_','CFG_'),''))
    
    
