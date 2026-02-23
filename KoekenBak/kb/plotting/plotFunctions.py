import pylab as pl
import os
import types
import numpy as np
#from scipy import array, zeros
#from scipy import argmax

from kb.tools.io import DataIO
from kb.modeling.tools import CodeIO
from kb.modeling.tools import AnalyseIO
from kb.plotting.objects import PlottingSession
from kb.plotting import Plotting2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.pyplot as p
from matplotlib import rc
rc('text', usetex=True)


def _reaction_label(rates, num):
    """Build a human-readable reaction string from the rates dict,
    falling back to the raw reaction number if rates is None or missing."""
    if rates is None:
        return str(num)
    key = str(num)
    if key not in rates:
        return key
    r = rates[key]
    reactants = ' + '.join(sp for sp in r['REACTANTS'] if sp)
    products  = ' + '.join(sp for sp in r['PRODUCTS']  if sp)
    return '{} $\\rightarrow$ {}'.format(reactants, products)


def plotProductionReactions(radii,preact,component,m,title,pfn,cfg,rates=None):
    """
    Plot the fractional production rate of the main reactions at each radius
    for a certain molecule.

    @keyword radii: Radii of the model calculation
    @type radii: list
    @keyword preact: Main production reactions + fractions at each radius.
    @type preact: dict()
    @keyword component: Get info of major or minor component calculation
    @type component: str
    @keyword m: Molecule of interest
    @type m: str
    @keyword pfn: Plotting filname
    @type pfn: str
    @keyword cfg_dict: Dictionary of cfg file
    @type cfg_dict: dict()
    """                                    
    cfg_dict = Plotting2.readCfg(cfg)
    if 'xmin' in cfg_dict:
        xmin=cfg_dict.pop('xmin')
    else:
        xmin=radii[0]
    if 'xmax' in cfg_dict:
        xmax=cfg_dict.pop('xmax')
    else:
        xmax=radii[-1]	
    if 'figsize' in cfg_dict:
        figsize = cfg_dict.pop('figsize')
    else:
        figsize = (12.5,8.5)
        
    names = AnalyseIO.getMainReactionNames(preact)	
    P = len(names)	
    #- Make colors and markers (of equal length as P)
    colors = mpl.cm.gist_rainbow(np.linspace(0, 1, P))
    markers = ['o','v','x','h','+','d','1']
    orig = markers
    for ii in range(P-len(markers)):
        markers.append(orig[ii])
        
    reactions = dict()
    for n in names:
        reactions[n] = []

    for ii,pro in preact.items():
        for pr in pro:
            reactions[pr[0]].append([ii,pr[1]])	
    p.figure(1,figsize=figsize)
    ax = p.subplot(111)
    ax.set_xscale('log')
    if 'xlogscale' in cfg_dict:
        if cfg_dict['xlogscale']==1:
            ax.set_xscale('log')
        if cfg_dict['xlogscale']==0:
            ax.set_xscale('linear')

    j = 0
    for ii,pr in reactions.items():			
        x = [radii[q[0]] for q in pr]
        y = [float(q[1]) for q in pr]
        p.plot(x,y,label=_reaction_label(rates,ii),color=colors[j],marker=markers[j])
        j += 1
    p.legend(fontsize=11)
    p.title(title)
    p.xlim((xmin,xmax))
    p.xlabel('Radius (cm)')
    p.ylabel('Fraction of total '+m+' produced')
    p.savefig(pfn+'.pdf',bbox_inches='tight')
    #pdf.savefig(bbox_inches='tight')
    print(pfn+'.pdf')
    p.clf()
    p.cla()



def plotProductionReactionsAccuracy(radii,preact,component,m,title,pfn,cfg,rates=None):
    """
    Plot the fractional production rate of the main reactions at each radius
    for a certain molecule.

    @keyword radii: Radii of the model calculation
    @type radii: list
    @keyword preact: Main production reactions + fractions at each radius.
    @type preact: dict()
    @keyword component: Get info of major or minor component calculation
    @type component: str
    @keyword m: Molecule of interest
    @type m: str
    @keyword pfn: Plotting filname
    @type pfn: str
    @keyword cfg_dict: Dictionary of cfg file
    @type cfg_dict: dict()
    """                                    
    cfg_dict = Plotting2.readCfg(cfg)
    if 'xmin' in cfg_dict:
        xmin=cfg_dict.pop('xmin')
    else:
        xmin=radii[0]
    if 'xmax' in cfg_dict:
        xmax=cfg_dict.pop('xmax')
    else:
        xmax=radii[-1]	
    if 'figsize' in cfg_dict:
        figsize = cfg_dict.pop('figsize')
    else:
        figsize = (12.5,8.5)
        
    names = AnalyseIO.getMainReactionNames(preact)	
    P = len(names)	
    
    #- Make colors and markers (of equal length as P)
    colours = mpl.cm.gist_rainbow(np.linspace(0, 1, P))
    colors = dict()
    colors['A'] = 'green'
    colors['B'] = 'orange'
    colors['C'] = 'red'
    colors['D'] = 'darkred'
    colors['E'] = 'k'
    markers = ['o','v','s','^','H','D','>','8']
    orig = markers
    for ii in range(P-len(markers)):
        markers.append(orig[ii])
        
    reactions = dict()
    for n in names:
        reactions[n] = []

    for ii,pro in preact.items():
        for pr in pro:
            reactions[pr[0]].append([ii,pr[1],pr[2]])	
            
            
    p.figure(1,figsize=figsize)
    ax = p.subplot(111)
    ax.set_xscale('log')
    if 'xlogscale' in cfg_dict:
        if cfg_dict['xlogscale']==1:
            ax.set_xscale('log')
        if cfg_dict['xlogscale']==0:
            ax.set_xscale('linear')

    j = 0
    for ii,pr in reactions.items():	
        x = [radii[q[0]] for q in pr]
        y = [float(q[1]) for q in pr]
        
        clrs = [colors[q[2]] for q in pr]
        #p.plot(x,y,label=ii,color=colours[j],marker=markers[j],markeredgecolor=colors[q[2]],markerfacecolor=colors[q[2]])
        p.plot(x,y,label=ii,color=colours[j],marker=markers[j],markerfacecolor=colors[q[2]])
        j += 1
    p.legend(fontsize=11)
    p.title(title)
    p.xlim((xmin,xmax))
    p.xlabel('Radius (cm)')
    p.ylabel('Fraction of total '+m+' produced')
    p.savefig(pfn+'.pdf',bbox_inches='tight')
    #pdf.savefig(bbox_inches='tight')
    print(pfn+'.pdf')
    p.clf()
    p.cla()




            
def plotDestructionReactions(radii,dreact,component,m,title,pfn,cfg,rates=None):
    """
    Plot the fractional destruction rate of the main reactions at each radius
    for a certain molecule.

    @keyword radii: Radii of the model calculation
    @type radii: list
    @keyword preact: Main destruction reactions + fractions at each radius.
    @type dreact: dict()
    @keyword component: Get info of major or minor component calculation
    @type component: str
    @keyword m: Molecule of interest
    @type m: str
    @keyword pfn: Plotting filname
    @type pfn: str
    @keyword cfg_dict: Dictionary of cfg file
    @type cfg_dict: dict()
    """
    cfg_dict = Plotting2.readCfg(cfg)
    if 'xmin' in cfg_dict:
        xmin=cfg_dict.pop('xmin')
    else:
        xmin=radii[0]
    if 'xmax' in cfg_dict:
        xmax=cfg_dict.pop('xmax')
    else:
        xmax=radii[-1]	
    if 'figsize' in cfg_dict:
        figsize = cfg_dict.pop('figsize')
    else:
        figsize = (12.5,8.5)
        
    names = AnalyseIO.getMainReactionNames(dreact)	
    P = len(names)	

    #- Make colors and markers (of equal length as P)
    colors = mpl.cm.gist_rainbow(np.linspace(0, 1, P))
    markers = ['o','v','x','h','+','d','1']
    orig = markers
    for ii in range(P-len(markers)):
        markers.append(orig[ii])
        
    reactions = dict()
    for n in names:
        reactions[n] = []

    for ii,pro in dreact.items():
        for pr in pro:
            reactions[pr[0]].append([ii,pr[1]])	
    p.figure(1,figsize=figsize)
    ax = p.subplot(111)
    ax.set_xscale('log')
    if 'xlogscale' in cfg_dict:
        if cfg_dict['xlogscale']==1:
            ax.set_xscale('log')
        if cfg_dict['xlogscale']==0:
            ax.set_xscale('linear')

    j = 0
    for ii,pr in reactions.items():			
        x = [radii[q[0]] for q in pr]
        y = [float(q[1]) for q in pr]
        p.plot(x,y,label=_reaction_label(rates,ii),color=colors[j],marker=markers[j])
        j += 1
    p.legend(fontsize=11)
    p.title(title)
    p.xlim((xmin,xmax))
    p.xlabel('Radius (cm)')
    p.ylabel('Fraction of total '+m+' destroyed')
    p.savefig(pfn+'.pdf',bbox_inches='tight')
    #pdf.savefig(bbox_inches='tight')
    print(pfn+'.pdf')
    p.clf()
    p.cla()



def plotDestructionReactionsAccuracy(radii,dreact,component,m,title,pfn,cfg,rates=None):
    """
    Plot the fractional destruction rate of the main reactions at each radius
    for a certain molecule.

    @keyword radii: Radii of the model calculation
    @type radii: list
    @keyword preact: Main destruction reactions + fractions at each radius.
    @type dreact: dict()
    @keyword component: Get info of major or minor component calculation
    @type component: str
    @keyword m: Molecule of interest
    @type m: str
    @keyword pfn: Plotting filname
    @type pfn: str
    @keyword cfg_dict: Dictionary of cfg file
    @type cfg_dict: dict()
    """
    cfg_dict = Plotting2.readCfg(cfg)
    if 'xmin' in cfg_dict:
        xmin=cfg_dict.pop('xmin')
    else:
        xmin=radii[0]
    if 'xmax' in cfg_dict:
        xmax=cfg_dict.pop('xmax')
    else:
        xmax=radii[-1]	
    if 'figsize' in cfg_dict:
        figsize = cfg_dict.pop('figsize')
    else:
        figsize = (12.5,8.5)
        
    names = AnalyseIO.getMainReactionNames(dreact)	
    P = len(names)	

    #- Make colors and markers (of equal length as P)
    colours = mpl.cm.gist_rainbow(np.linspace(0, 1, P))
    colors = dict()
    colors['A'] = 'green'
    colors['B'] = 'orange'
    colors['C'] = 'red'
    colors['D'] = 'darkred'
    colors['E'] = 'k'
    markers = ['o','v','s','^','H','D','>','8']
    orig = markers
    for ii in range(P-len(markers)):
        markers.append(orig[ii])
        
    reactions = dict()
    for n in names:
        reactions[n] = []

    for ii,pro in dreact.items():
        for pr in pro:
            reactions[pr[0]].append([ii,pr[1],pr[2]])	
    p.figure(1,figsize=figsize)
    ax = p.subplot(111)
    ax.set_xscale('log')
    if 'xlogscale' in cfg_dict:
        if cfg_dict['xlogscale']==1:
            ax.set_xscale('log')
        if cfg_dict['xlogscale']==0:
            ax.set_xscale('linear')

    j = 0
    for ii,pr in reactions.items():			
        x = [radii[q[0]] for q in pr]
        y = [float(q[1]) for q in pr]
        clrs = [colors[q[2]] for q in pr]
        
        #p.plot(x,y,label=ii,color=colours[j],marker=markers[j],markeredgecolor=colors[q[2]],markerfacecolor=colors[q[2]])
        p.plot(x,y,label=_reaction_label(rates,ii),color=colours[j],marker=markers[j],markerfacecolor=colors[q[2]])
        j += 1
    p.legend(fontsize=11)
    p.title(title)
    p.xlim((xmin,xmax))
    p.xlabel('Radius (cm)')
    p.ylabel('Fraction of total '+m+' destroyed')
    p.savefig(pfn+'.pdf',bbox_inches='tight')
    #pdf.savefig(bbox_inches='tight')
    print(pfn+'.pdf')
    p.clf()
    p.cla()





