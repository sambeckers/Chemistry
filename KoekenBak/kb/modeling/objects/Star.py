# -*- coding: utf-8 -*-

"""
Module including functions for stellar parameters, and the STAR class and 
its methods and attributes.

Author: R. Lombaert

"""

import types
from glob import glob
import os
#from scipy import pi, log, sqrt
#from scipy import array, exp, zeros
#from scipy import integrate, linspace
#from scipy import argmin,argmax, empty
#from scipy.interpolate import interp1d
import operator
#from numpy import savetxt
from astropy import units as u
from astropy import constants as cst

import kb.path
#from kb.tools.units import Equivalency as eq
from kb.tools.io import Database
from kb.tools.io import DataIO


def getStar(star_grid,modelid,idtype='Chemistry'):
    
    '''
    Grab a Star() object from a list of such objects, given a model id.
    
    If no modelid is found, an empty list is returned. If Star() objects are 
    found (even only one), a list of them is returned.
    
    Based on the cooling modelid.
    
    @param star_grid: the Star() objects
    @type star_grid: list[Star()]
    @param modelid: the given modelid for which the selection is made.
    @type modelid: string
    
    @keyword idtype: The type of model id
                    
                     (default: GASTRONOOM)    
    @type idtype: string

    @return: The models matching the modelid
    @rtype: list[Star()]
    
    '''
    
    modelid, idtype = str(modelid), str(idtype)
    return [s for s in star_grid if s['LAST_%s_MODEL'%idtype] == modelid]
    
    
    
def makeStars(models,id_type,path,code):
    
    '''
    Make a list of dummy Star() objects.

    @param models: model_ids for the new models
    @type models: list[string]
    @param id_type: The type of id (PACS, GASTRONOOM, MCMAX)
    @type id_type: string
    @param path: Output folder in the code's home folder
    @type path: string
    @param code: The code (which is not necessarily equal to id_type, such as 
                 for id_type == PACS)
    @type code: string

    @return: The parameter sets, mostly still empty!
    @rtype: list[Star()]
    
    '''
    
    extra_pars = dict([('path_'+code.lower(),path)])
    star_grid = [Star(example_star={'LAST_%s_MODEL'%id_type.upper():model},\
                      **extra_pars) 
                 for model in models]
    return star_grid
      

    
class Star(dict):
    
    """
    Star class maintains information about a stellar model and its properties.

    Inherits from dict.
    
    """



    def __init__(self,path_chemistry='',extra_input=None,\
                 example_star=dict()):
        
        """
        Initiate an instance of the STAR class.
        
        @keyword path_chemistry: path in ~/Chemistry/ for modeling out/input
                                  
                                  (default: None)
        @type path_chemistry: string
        @keyword example_star: if not None the STAR object is exact duplicate 
                               of example_star. Can be a normal dictionary as 
                               well. Paths are not copied and need to be given 
                               explicitly.
                                    
                               (default: None)
        @type example_star: dict or Star()                                  
        @keyword extra_input: extra input that you wish to add to the dict
        
                              (default: None)
        @type extra_input: dict or Star()
        
        @return: STAR object in the shape of a dictionary which includes all 
                 stellar data available, if None are passed for both options it
                 is an empty dictionary; if an example star is passed it has a 
                 dict that is an exact duplicate of the example star's dict
        @rtype: Star()
        
        """    
            
        super(Star, self).__init__(example_star)
        if not extra_input is None: self.update(extra_input)
        self.Rsun = cst.R_sun.cgs.value         #in cm  Harmanec & Prsa 2011
        self.Msun = 1.98547e33      #in g   Harmanec & Prsa 2011
        self.Mearth = cst.M_earth.cgs.value   # in g
        self.Tsun = 5779.5747            #in K   Harmanec & Psra 2011
        self.Lsun = cst.L_sun.cgs.value           #in erg/s
        self.au = 149598.0e8             #in cm
        self.c = cst.c.cgs.value          #in cm/s
        self.h = cst.h.cgs.value         #in erg*s, Planck constant
        self.k = cst.k_B.cgs.value          #in erg/K, Boltzmann constant
        self.sigma = cst.sigma_sb.cgs.value         #in erg/cm^2/s/K^4  = g / (K^4 s^3) Stefan_boltzmann constant
        self.mh = cst.m_p.cgs.value           #in g, mass hydrogen atom
        self.G = cst.G.cgs.value           # in cm^3 g^-1 s^-2
        
        self.path_chemistry = path_chemistry   

        #-- Convenience paths
        kb.path.cout = os.path.join(kb.path.chemistry,self.path_chemistry)
        
        
        

    def __getitem__(self,key):

        """
        Overriding the standard dictionary __getitem__ method.
        
        @param key: Star()[key] where key is a string for which a corresponding
                    dictionary value is searched. If the key is not present in 
                    the dictionary, an attempt is made to calculate it from 
                    already present data; if it fails a KeyError is still 
                    raised. 
        @type key: string            
        
        @return: The value from the Star() dict for key
        @rtype: any
        
        """
        
        #if not self.has_key(key):
            #self.missingInput(key)
            #return super(Star,self).__getitem__(key)
        #elif super(Star,self).__getitem__(key) == '%':
            #del self[key]
            #self.missingInput(key)
            #value = super(Star,self).__getitem__(key)
            #self[key] = '%'
            #return value 
        #else:
            #return super(Star,self).__getitem__(key)

        if key not in self:
            self.missingInput(key)
            return super(Star,self).__getitem__(key)
        elif super(Star,self).__getitem__(key) == '%':
            del self[key]
            self.missingInput(key)
            value = super(Star,self).__getitem__(key)
            self[key] = '%'
            return value 
        else:
            return super(Star,self).__getitem__(key)


    def __cmp__(self,star):
        
        """
        Overriding the standard dictionary __cmp__ method.
        
        A parameter set (dictionary of any type) is compared with this instance
        of Star(). 
        
        An attempt is made to create keys with values in each dict, if the 
        other has keys that are not present in the first. If this fails, False
        is returned.     
        
        @param star: A different parameter set. 
        @type star: dict or Star()             
        
        @return: The comparison between this object and star
        @rtype: bool
        
        """
        
        try:
            all_keys = set(list(self.keys()) + list(star.keys()))
            for k in all_keys:
                #if not self.has_key(): 
                if k not in self: 
                    self[k]
                #if not star.has_key():
                if k not in star:
                    star[k]
        except KeyError:
            print('Comparison error: Either STAR1 or STAR2 contains a key ' + \
                  'that cannot be initialized for the other.')
            print('Both STAR instances are considered to be unequal.')
        finally:
            if isinstance(star, super(Star)):
                return cmp(super(Star,self), super(Star,star))
            else:
                return cmp(super(Star,self), star)                 



    def missingInput(self,missing_key):
        
        """
        Try to resolve a missing key.
                
        @param missing_key: the missing key for which an attempt will be made 
                            to calculate its value based on already present 
                            parameters
        @type missing_key: string
        
        """
        
        if missing_key in ('T_STAR','L_STAR','R_STAR'):
            self.calcTLR()
        else:
            pass
            
            

    def calcTLR(self):  
        
        """
        Stefan-Boltzmann's law.
            
        Star() object needs to have at least 2 out of 3 parameters (T,L,R), 
        with in L and R in solar values and T in K.
    
        The one missing parameter is calculated. 
    
        This method does nothing if all three are present.
        
        """
        
        if not self.has_key('T_STAR'):
            self['T_STAR']=(float(self['L_STAR'])/float(self['R_STAR'])**2.)\
                                **(1/4.)*self.Tsun
        elif not self.has_key('L_STAR'):
            self['L_STAR']=(float(self['R_STAR']))**2.*\
                                (float(self['T_STAR'])/self.Tsun)**4.
        elif not self.has_key('R_STAR'):
            self['R_STAR']=(float(self['L_STAR'])*\
                                (self.Tsun/float(self['T_STAR']))**4)**(1/2.)
        else:
            pass 

