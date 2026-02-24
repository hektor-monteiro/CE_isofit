#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 08:40:17 2017

Provides usefull function to deal with GAIA data

28/01/2021 - started Gaia eDR3 revidion

@author: hmonteiro
"""

import numpy as np
from scipy.interpolate import interp1d,LinearNDInterpolator,griddata
from scipy.integrate import trapezoid,cumulative_trapezoid
import matplotlib.pyplot as plt
from astropy import units as u
import sys
from scipy import stats
#from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import cdist
from scipy import stats
from astropy.modeling import models

###############################################
# function to add columns to rec array

def add_col(array,col,col_name):
    col_type=(col_name, col.dtype)
    y=np.zeros(array.shape, dtype=array.dtype.descr+[col_type])    
    for name in array.dtype.names: y[name] = array[name]    
    y[col_type[0]]=col    
    return y

###############################################
# Save binary file with full isochrone grid
def save_mod_grid(dir, isoc_set='UBVRI'):
    if(isoc_set == 'UBVRI'):
        mod_grid = np.genfromtxt(dir+'parsec1.2sc-full.txt',names=True)
        np.save(dir+'full_isoc_UBVRI.npy',mod_grid)
        
    if(isoc_set == 'GAIA'):
        mod_grid = np.genfromtxt(dir+'parsec1.2s-tycho-full.txt',names=True)
        use = ['Zini','Age','Mini','Gmag','G_BPmag','G_RPmag','B_Tmag',
               'V_Tmag','Jmag','Hmag','Ksmag']
        np.save(dir+'full_isoc_GAIA.npy',mod_grid[use])
        
        
    return

###############################################
# Load binary file with full isochrone grid
# and returns array of data and arrays of unique age and Z values
#
def load_mod_grid(dir, isoc_set='UBVRI'):
    global mod_grid
    global age_grid
    global z_grid

    if(isoc_set == 'UBVRI'):
        mod_grid = np.load(dir+'full_isoc_UBVRI.npy')
        
    if(isoc_set == 'GAIA'):
        mod_grid = np.load(dir+'full_isoc_GAIA_CMD33.npy')

    if(isoc_set == 'GAIA_eDR3'):
        mod_grid = np.load(dir+'full_isoc_Gaia_eDR3_CMD34.npy')

    if(isoc_set == 'MIST-UBVRI'):
        mod_grid = np.load(dir+'full_isoc_MIST-UBVRI.npy')
        
    if(isoc_set == 'MIST-GAIA'):
        mod_grid = np.load(dir+'full_isoc_MIST-GAIA.npy')
            
    age_grid = np.unique(mod_grid['logAge'])
    z_grid = np.unique(mod_grid['Zini'])
    
    return mod_grid, age_grid, z_grid
###############################################
# truncated PAreto for Salpeter IMF

def salpeter(alpha, nstars, Mmin, Mmax,seed=None):
    
    mass_int = np.flip(np.logspace(np.log10(Mmax),np.log10(Mmin), 1000),axis=0)

    ind_low = np.where(mass_int <= 1.)    
    imf_val = mass_int**(-alpha)
    
    #normalize
    imf_norm =  imf_val / (trapezoid(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumulative_trapezoid(imf_norm,mass_int, initial=0)
    
    np.random.seed(seed)
    
    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(np.random.rand(nstars))
    return gen_masses

###############################################
# tapered deMarchi IMF

def deMarchi(alpha, beta, nstars, Mmin, Mmax,seed=None):
    
    mass_int = np.flip(np.logspace(np.log10(Mmax),np.log10(Mmin), 1000),axis=0)

    ind_low = np.where(mass_int <= 1.)    
    imf_val = mass_int**(-alpha)*(1.-np.exp(-(mass_int/1.)**(-beta)))
    
    #normalize
    imf_norm =  imf_val / (trapezoid(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumulative_trapezoid(imf_norm,mass_int, initial=0)
    
    np.random.seed(seed)
    
    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(np.random.rand(nstars))
    return gen_masses

###############################################
#     
def MillerScalo(alpha,nstars, Mmin, Mmax,seed=None):
    
    mass_int = np.flip(np.logspace(np.log10(Mmax),np.log10(Mmin), 100),axis=0)

    ind_low = np.where(mass_int <= 1.)    
    imf_val = mass_int**(-alpha)
    imf_val[ind_low] = mass_int[ind_low]**(0.)
    
    #normalize
    imf_norm =  imf_val / (trapezoid(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumulative_trapezoid(imf_norm,mass_int, initial=0)
    
    np.random.seed(seed)
    
    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(np.random.rand(nstars))
    return gen_masses

###############################################
# Chabrier (2001) exponential form of the IMF.
# http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
    
def chabrier(alpha,nstars, Mmin, Mmax,seed=None):
    
    #mass_int = np.linspace(Mmin,Mmax,1000)
    mass_int = np.flip(np.logspace(np.log10(Mmax),np.log10(Mmin),1000),axis=0)
    # Chabrier (2001) exponential form of the IMF.
    # http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
    
#    imf_val = 3. * mass_int ** (-3.3) * np.exp(-(716.4 / mass_int) ** 0.25)
    
    #Chabrier (2003) - http://adsabs.harvard.edu/abs/2003PASP..115..763C
    ind_low = np.where(mass_int <= 1.)    
    imf_val = 4.43e-2*mass_int**(-alpha)
    imf_val[ind_low] = 0.158*np.exp(-0.5*(np.log10(mass_int[ind_low])-
           np.log10(0.08))**2/0.69**2)
    
    #normalize
    imf_norm =  imf_val / (trapezoid(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumulative_trapezoid(imf_norm,mass_int, initial=0)
    
    r = np.random.RandomState(seed)

    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(r.rand(nstars))

    return gen_masses

###############################################
# Chabrier (2001) exponential form of the IMF.
# http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
    
def chabrier_bin(nstars, Mmin, Mmax,seed=None):
    
    mass_int = np.linspace(Mmin,Mmax,100)
#    mass_int = np.logspace(np.log10(Mmax),np.log10(Mmin),base=10.)
    # Chabrier (2001) exponential form of the IMF.
    # http://adsabs.harvard.edu/abs/2001ApJ...554.1274C
    
#    imf_val = 3. * mass_int ** (-3.3) * np.exp(-(716.4 / mass_int) ** 0.25)
    
    #Chabrier (2003) - http://adsabs.harvard.edu/abs/2003PASP..115..763C
    ind_low = np.where(mass_int <= 1.)    
    imf_val = 4.43e-2*mass_int**(-2.1)
    imf_val[ind_low] = 0.086*np.exp(-0.5*(np.log10(mass_int[ind_low])-
           np.log10(0.22))**2/0.57**2)
    
    #normalize
    imf_norm =  imf_val / (trapezoid(imf_val,mass_int))

    # get cumulative distribution
    cum_imf = cumulative_trapezoid(imf_norm,mass_int, initial=0)
    
    np.random.seed(seed)
    
    # sample from IMF
    gen_masses = (interp1d(cum_imf,mass_int))(np.random.rand(nstars))
    return gen_masses


###############################################
# sample from given isochrone
    
def sample_from_isoc(rawisoc,bands,refMag,nstars,imf='chabrier',alpha=2.3,
                     beta=-3,Mcut=False,seed=None,binmasses=None):
    
#    if (Mcut and Abscut): sys.exit('Both Mcut and Abscut are not allowed simultaneaously!')
    
    if(Mcut):
        ind = np.abs(rawisoc[refMag]-Mcut).argmin()
        Mmin, Mmax = rawisoc['Mini'][ind], np.max(rawisoc['Mini'])
        if (Mmin==Mmax): Mmin=0.99*Mmin
    else:
        Mmin, Mmax = np.min(rawisoc['Mini']),np.max(rawisoc['Mini'])
        
    # get mass vector according to IMF
    
    if imf:
        if imf == 'salpeter':
            # print 'Using Salpeter IMF with alpha=2.35 in mass interval: [',Mmin,',',Mmax,']'
            masses = salpeter(alpha, nstars, Mmin, Mmax)
        if imf == 'chabrier':
            # print 'Using Chabrier(2001) IMF in mass interval: [',Mmin,',',Mmax,']'
            masses = chabrier(alpha,nstars, Mmin, Mmax,seed=seed)
        if imf == 'MillerScalo':
            # print MillerScalo
            masses = MillerScalo(alpha,nstars, Mmin, Mmax,seed=seed)
        if imf == 'deMarchi':
            masses = deMarchi(alpha,beta, nstars, Mmin, Mmax,seed=seed)
    else:
        raise ValueError('The IMF was not specified')

    if (binmasses is not None):
        masses = binmasses
        masses[masses<Mmin]=Mmin
        masses[masses>Mmax]=Mmax

    # interpolate photometry on the grid for the given masses
    photint = []

    for filter in bands:
        aux = interp1d(rawisoc['Mini'],rawisoc[filter])
        photint.append(aux(masses))
        
    # get final masses
    aux = interp1d(rawisoc['Mini'],rawisoc['Mass'])
    finalmasses = aux(masses)
     
                
    photint.append(masses)
    photint.append(finalmasses)
    
    cols = bands[:]
    cols.append('Mini')
    cols.append('Mass')
    
    return np.core.records.fromarrays(photint, names=cols)
    
###############################################
# function to get isochrone from grid given an age, metalicity

def get_iso_from_grid(age,met,bands,refMag,Abscut=False, nointerp=False):
    
    global mod_grid, age_grid, z_grid
    # check to see if grid is loaded
    if 'mod_grid' not in globals(): 
        raise NameError('Isochrone grid not loaded!')
        
    # find closest values to given age and Z
    dist_age = np.abs(age - age_grid)#/age
    ind_age = dist_age.argsort()
    dist_z = np.abs(met - z_grid)#/met
    ind_z = dist_z.argsort()
    
    dist0 = np.sqrt(dist_age[ind_age[0]]**2 + dist_z[ind_z[0]]**2)
    dist1 = np.sqrt(dist_age[ind_age[1]]**2 + dist_z[ind_z[1]]**2)

#    dist0 = np.sqrt(dist_age[ind_age[0]]**2)
#    dist1 = np.sqrt(dist_age[ind_age[1]]**2)
    
    dist_age_0 = dist_age[ind_age[0]]/(dist_age[ind_age[0]]+dist_age[ind_age[1]])
    dist_age_1 = dist_age[ind_age[1]]/(dist_age[ind_age[0]]+dist_age[ind_age[1]])
    dist_z_0 = dist_z[ind_z[0]]/(dist_z[ind_z[0]]+dist_z[ind_z[1]])
    dist_z_1 = dist_z[ind_z[1]]/(dist_z[ind_z[0]]+dist_z[ind_z[1]])
    
    dist0 = np.sqrt(dist_age_0**2 + dist_z_0**2)
    dist1 = np.sqrt(dist_age_1**2 + dist_z_1**2)
    
    # get the closest isochrone to the given age and Z
    #apply absolute mag cut if set
    if(Abscut):
        iso1 = mod_grid[(mod_grid['logAge'] == age_grid[ind_age[0]]) & 
                       (mod_grid['Zini'] == z_grid[ind_z[0]]) & 
                       (mod_grid[refMag] < Abscut)]
        iso2 = mod_grid[(mod_grid['logAge'] == age_grid[ind_age[1]]) & 
                       (mod_grid['Zini'] == z_grid[ind_z[1]]) & 
                       (mod_grid[refMag] < Abscut)]
    else:
        iso1 = mod_grid[(mod_grid['logAge'] == age_grid[ind_age[0]]) &
                       (mod_grid['Zini'] == z_grid[ind_z[0]])]
        iso2 = mod_grid[(mod_grid['logAge'] == age_grid[ind_age[1]]) &
                       (mod_grid['Zini'] == z_grid[ind_z[1]])]   
        
    photint = []
    
    for filter in bands:
        mass_int = []
        finalmass_int = []
        f_int = []
        
        for n in np.unique(iso1['label']):
            
            f1 = iso1[filter][iso1['label'] == n]
            f2 = iso2[filter][iso2['label'] == n]
            
            m1 = iso1['Mini'][iso1['label'] == n]
            m2 = iso2['Mini'][iso2['label'] == n]

            mf1 = iso1['Mass'][iso1['label'] == n]
            mf2 = iso2['Mass'][iso2['label'] == n]

            if(f1.size < 2 or f2.size < 2):
                    
                continue

            elif(f1.size > f2.size):
                npoints = f2.size
                
                f1i = interp1d(np.arange(f1.size),f1)
                f1 = f1i(np.linspace(0,f1.size-1,npoints))
                
                m1i = interp1d(np.arange(m1.size),m1)
                m1 = m1i(np.linspace(0,m1.size-1,npoints))
                
                mf1i = interp1d(np.arange(mf1.size),mf1)
                mf1 = m1i(np.linspace(0,mf1.size-1,npoints))
                
            else:
                npoints = f1.size

                f2i = interp1d(np.arange(f2.size),f2)
                f2 = f2i(np.linspace(0,f2.size-1,npoints))
                
                m2i = interp1d(np.arange(m2.size),m2)
                m2 = m2i(np.linspace(0,m2.size-1,npoints))
                
                mf2i = interp1d(np.arange(mf2.size),mf2)
                mf2 = mf2i(np.linspace(0,mf2.size-1,npoints))
                
            t = dist0/(dist0+dist1)
            
            mass_int = np.concatenate([mass_int, (1.-t)*m1+t*m2])
            finalmass_int = np.concatenate([finalmass_int, (1.-t)*mf1+t*mf2])
            f_int = np.concatenate([f_int, (1.-t)*f1+t*f2 ])
            

        photint.append(f_int)

    # keep mass field for future use
    photint.append(mass_int)
    photint.append(finalmass_int)

##########################################################
    if nointerp:
        # get the closest isochrone to the given age and Z
        #apply absolute mag cut if set
        if(Abscut):
            iso = mod_grid[(mod_grid['logAge'] == age_grid[ind_age[0]]) & 
                           (mod_grid['Zini'] == z_grid[ind_z[0]]) & 
                           (mod_grid[refMag] < Abscut)]
        else:
            iso = mod_grid[(mod_grid['logAge'] == age_grid[ind_age[0]]) &
                          (mod_grid['Zini'] == z_grid[ind_z[0]])]
            
        photint = []

        for filter in bands:
            photint.append(iso[filter])
        photint.append(iso['Mini'])
        photint.append(iso['Mass'])
###########################################################
        
    
    cols = bands[:]
    cols.append('Mini')
    cols.append('Mass')
    
    return np.core.records.fromarrays(photint, names=cols)
    
###############################################
# function to get CCM model coeficients
# for the GAIA filters we used: https://arxiv.org/pdf/1008.0815.pdf

def ccm_coefs(band):
    # CCM coefficients revised by O'Donnell (1994)
    dict_94 = {'Bmag': [4460.62, 0.9999253931841896, 0.94553192328962365],
            'Hmag': [16369.53, 0.2596053235545497, -0.23834844166071026],
            'Imag': [8036.57, 0.76735566136864775, -0.51126852210308293],
            'Jmag': [12314.46, 0.4105283397212145, -0.37691364988341475],
            'Kmag': [21937.18, 0.16203573610362962, -0.14876800161430803],
            'Rmag': [6557.09, 0.90991266273182836, -0.29970863780329793],
            'Umag': [3641.89, 0.96420188342499813, 1.784213363585738],
            'Vmag': [5501.7, 0.99974902186052628, -0.0046292182005786527],
            'B_Tmag': [4350.0, 1.0017962252392263, 1.0362801277999945],
            'V_Tmag': [5050.0, 1.0044713895752799, 0.36691338631832937],
            'G_BPmag': [5320.0, 1.0042005025756102, 0.12595552528038209],
            'G_RPmag': [7970.0, 0.77261277538121242, -0.49646921986943959],
            'Gmag': [6730.0, 0.89044911179059461, -0.31133471454740169]}
    
    # Original Cardelli, Clayton, and Mathis (1989 ApJ. 345, 245)
    dict_ccm_ori = {'Bmag': [4460.62, 1.0025759394309195, 0.92908077063668137],
                    'Hmag': [16369.53, 0.2596053235545497, -0.23834844166071026],
                    'Imag': [8036.57, 0.77833673604251075, -0.57683220088641463],
                    'Jmag': [12314.46, 0.4105283397212145, -0.37691364988341475],
                    'Kmag': [21937.18, 0.16203573610362962, -0.14876800161430803],
                    'Rmag': [6557.09, 0.90937658249478737, -0.28122644122534407],
                    'Umag': [3641.89, 0.95941873606501926, 1.8578498871346709],
                    'Vmag': [5501.7, 0.99957590813034558, -0.0033509151940263101],
                    'B_Tmag': [4350.0, 0.99736986014984541, 1.0711315721870396],
                    'V_Tmag': [5050.0, 1.015771886403408, 0.28589221136305393],
                    'G_BPmag': [5320.0, 1.0087722119558464, 0.092674041614019612],
                    'G_RPmag': [7970.0, 0.78420684534377993, -0.56542798221110957],
                    'Gmag': [6730.0, 0.89312322728995797, -0.31352377819769739]}

    return dict_94[band]
   
###############################################

def gaia_ext_coefs(band):

    dict_gaia = {'G_BPmag': [1.1517,-0.0871,-0.0333,0.0173,-0.0230,0.0006,0.0043],
                 'G_RPmag': [0.6104,-0.0170,-0.0026,-0.0017,-0.0078,0.00005,0.0006],
                 'Gmag': [0.9761,-0.1704,0.0086,0.0011,-0.0438,0.0013,0.0099]}
    return dict_gaia[band]

###############################################
def gaia_ext_Hek(color, Av,band):
    
    Avv = color*0. + Av
    
    # polynomial values for FITZPATRICK & MASSA (2019) and Gaia eDR3 log g = 2,4 FeH=0 
    if (band == 'G_BPmag'):
        poly = models.Polynomial2D(degree=4)
        poly._parameters = np.array([ 1.04147486e+00,  2.01708670e-02,  3.64641123e-03, -2.63700516e-04,
        6.31474566e-05, -8.83945036e-02,  1.83825389e-02,  4.50477706e-03,
        3.26750519e-03, -1.62928523e-02, -2.04675757e-03, -5.40740309e-03,
        8.07038769e-04,  2.95228508e-03, -6.76541300e-04])
        coeffs = dict((name, poly._parameters[i]) for i, name in enumerate(poly.param_names))
        poly = models.Polynomial2D(degree=4,**coeffs)
        k = poly(Avv,color)
 
               
    if (band == 'G_RPmag'):
        poly = models.Polynomial2D(degree=4)
        poly._parameters = np.array([ 6.38099500e-01,  2.30574591e-03, -8.99468273e-04,  4.46199671e-04,
        6.24722092e-05, -2.25374369e-02, -1.02043689e-02, -3.76323711e-03,
        3.50246307e-03,  6.30231188e-03,  6.40243260e-03, -5.49766107e-03,
       -3.10851841e-03,  3.09564948e-03, -7.33372566e-04])
        coeffs = dict((name, poly._parameters[i]) for i, name in enumerate(poly.param_names))
        poly = models.Polynomial2D(degree=4,**coeffs)
        k = poly(Avv,color)

    
    if (band == 'Gmag'):
        poly = models.Polynomial2D(degree=4)
        poly._parameters = np.array([ 8.20652902e-01,  2.61920418e-02,  2.68469208e-03,  9.36694336e-05,
       -2.52100168e-05, -1.27466265e-01,  1.93940175e-02,  1.88492527e-03,
       -9.17877230e-04, -1.38032911e-02,  2.02177729e-03,  7.35239495e-04,
       -1.19970676e-03, -5.13235046e-04,  2.08371649e-04])
        coeffs = dict((name, poly._parameters[i]) for i, name in enumerate(poly.param_names))
        poly = models.Polynomial2D(degree=4,**coeffs)
        k = poly(Avv,color)
            
        
    return k
    
###############################################
# Make an observed synthetic cluster given an isochrone,
# distance, E(B-V) and Rv
# Al/Av = a + b/rv
# Av = rv*ebv

def make_obs_iso(bands, iso, dist, Av, gaia_ext = False):
    #redden and move isochrone
    obs_iso = np.copy(iso)
    
    color = iso['G_BPmag'] - iso['G_RPmag']

    for filter in bands:
        
        if gaia_ext:
            
            # get coeficients
#            c1, c2, c3, c4, c5, c6, c7 = gaia_ext_coefs(filter)
#            AloAv = c1 + c2*color + c3*color**2 + c4*color**3 + c5*Av + c6*Av**2 + c7*color*Av
            
            AloAv = gaia_ext_Hek(color, Av,filter)
            
            # apply correction
            obs_iso[filter] = iso[filter] + 5.*np.log10(dist*1.e3) - 5.+ AloAv*Av
            
        else:
            # get CCm coeficients
            wav,a,b = ccm_coefs(filter)
        
            # apply ccm model and make observed iso
            obs_iso[filter] = iso[filter] + 5.*np.log10(dist*1.e3) - 5.+ ( (a + b/3.1)*Av )
        
    return obs_iso


    
###############################################
# add some binaries
# generate probability of being binary

def add_binaries(bin_frac,isoc,isoc_bin,bands,refMag,imf='chabrier',alpha=2.3,
                 beta=-3,seed=None,binratio=0.8):
    
    if(seed != None): 
        r = np.random.RandomState(seed+1)
    else:
        r = np.random.RandomState(seed)

    prob_bin = r.rand(isoc.size)
    nbin = isoc[prob_bin < bin_frac].size
    isoc = np.copy(isoc)
    isoc_bin = np.copy(isoc_bin)
    
#    masses = binratio*isoc['Mini'][prob_bin < bin_frac]
#    masses = r.beta(35,10.,nbin)*isoc['Mini'][prob_bin < bin_frac]
#    masses = r.rand(isoc.size)*isoc['Mini']
#    masses = r.uniform(0,1,nbin)*isoc['Mini'][prob_bin < bin_frac]


#    lower, upper = 0., 1.
#    mu, sigma = 1., 0.05
#    fac = stats.truncnorm((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma)    
#    masses = fac.rvs(nbin)*isoc['Mini'][prob_bin < bin_frac]
    
#    indlow = np.where(isoc['Mini'][prob_bin < bin_frac] < 1.5)
#    indhi = np.where(isoc['Mini'][prob_bin < bin_frac] >= 3.)
#    indmid = np.where((isoc['Mini'][prob_bin < bin_frac] >= 1.5) & (isoc['Mini'][prob_bin < bin_frac] < 3))
#    masses=np.zeros(nbin)
#    masses[indlow] = salpeter(0.01,indlow[0].size,0.01,1.,seed)*isoc['Mini'][prob_bin < bin_frac][indlow]
#    masses[indhi] = salpeter(0.15,indhi[0].size,0.01,1.,seed)*isoc['Mini'][prob_bin < bin_frac][indhi]
#    masses[indmid] = salpeter(0.45,indmid[0].size,0.01,1.,seed)*isoc['Mini'][prob_bin < bin_frac][indmid]
#
#    bin_comp_stars = sample_from_isoc(isoc_bin,bands,refMag,nbin,imf=imf,alpha=alpha,
#                                      beta=beta, seed=seed,binmasses=masses)

    bin_comp_stars = sample_from_isoc(isoc_bin,bands,refMag,nbin,imf=imf,alpha=alpha,
                                      beta=beta,seed=seed)
    

    for filter in bands:
        m1 = isoc[filter][prob_bin < bin_frac]
        m2 = bin_comp_stars[filter]
        mcomb = m1 - 2.5 * np.log10(1.0 + 10**(-0.4*(m2-m1)))
        isoc[filter][prob_bin < bin_frac] = mcomb
        
    return isoc



###############################################
# add some errors
# IMPORTANT: the error coeficients for J,H, K are just copies of Imag 
# since we dont have data for them

def add_phot_errors(isoc,bands):
    # gaia errors from https://www.cosmos.esa.int/web/gaia/science-performance
    
    er_isoc = np.copy(isoc)
    gaia_bands = ['Gmag','G_BPmag','G_RPmag']
    
    er_coefs = {'Umag': [0.03093, 3.93086E-13, 0.79563],
                'Bmag': [0.03078, 6.92477E-13, 0.84029],
                'Vmag': [0.02869, 4.70556E-12, 0.86897],
                'Rmag': [0.02400, 3.77483E-12, 0.82648],
                'Imag': [0.02831, 8.05872E-13, 0.73068],
                'Jmag': [0.02831, 8.05872E-13, 0.73068],
                'Hmag': [0.02831, 8.05872E-13, 0.73068],
                'Kmag': [0.02831, 8.05872E-13, 0.73068],
                'B_Tmag': [0.03078, 6.92477E-13, 0.84029],
                'V_Tmag': [0.02869, 4.70556E-12, 0.86897]}

                
    for filter in bands:
        
        if filter == 'Gmag':
            z=[]
            for mag in isoc[filter]:
                z.append(np.max([10**0.4*(12 - 15), 10**0.4*(mag - 15.)]))
            er = 1.e-3*np.sqrt(np.abs(0.04895*np.array(z)**2 + 1.8633*np.array(z) + 0.0001985))
                        
            er_isoc[filter] = isoc[filter] * np.random.normal(1., np.abs(er/isoc[filter]), isoc.size)
            
        elif (filter == 'G_BPmag' or filter == 'G_RPmag'):
            # WARNING !!!! this needs to be updated to correct formulas!!!
            z=[]
            for mag in isoc['Gmag']:
                z.append(np.max([10**0.4*(11 - 15), 10**0.4*(mag - 15.)]))
            er = 1.e-3*np.sqrt(np.abs(0.04895*np.array(z)**2 + 1.8633*np.array(z) + 0.0001985))
            er_isoc[filter] = isoc[filter] * np.random.normal(1., np.abs(er/isoc[filter]), isoc.size)
            
        else:
            er = er_coefs[filter][0] +  er_coefs[filter][1] * np.exp(isoc[filter]/er_coefs[filter][2])
            er[er > 0.5*3.] = 0.5*3.        
            er_isoc[filter] = isoc[filter] * np.random.normal(1., np.abs(er/isoc[filter])/3., isoc.size)
    
        
    return er_isoc

###############################################
# return the vector of errors
# IMPORTANT: the error coeficients for J,H, K are just copies of Imag 
# since we dont have data for them
# GAIA errors from: https://www.cosmos.esa.int/web/gaia/science-performance

def get_phot_errors(isoc,bands):
    
    errors = np.copy(isoc)
    
    er_coefs = {'Umag': [0.03093, 3.93086E-13, 0.79563],
                'Bmag': [0.03078, 6.92477E-13, 0.84029],
                'Vmag': [0.02869, 4.70556E-12, 0.86897],
                'Rmag': [0.02400, 3.77483E-12, 0.82648],
                'Imag': [0.02831, 8.05872E-13, 0.73068],
                'Jmag': [0.02831, 8.05872E-13, 0.73068],
                'Hmag': [0.02831, 8.05872E-13, 0.73068],
                'Kmag': [0.02831, 8.05872E-13, 0.73068],
                'B_Tmag': [0.03078, 6.92477E-13, 0.84029],
                'V_Tmag': [0.02869, 4.70556E-12, 0.86897]}
    
    
                
    for filter in bands:
        
        if filter == 'Gmag':
            z=[]
            for mag in isoc[filter]:
                z.append(np.max([10**(0.4*(12 - 15)), 10**(0.4*(mag - 15.))]))
            er = 1.e-3*np.sqrt(0.04895*np.array(z)**2 + 1.8633*np.array(z) + 0.0001985)
            isoc = add_col(isoc,er,'e_Gmag')
            
        elif (filter == 'G_BPmag'):
            color = isoc['G_BPmag'] - isoc['G_RPmag'] 
            aBP	=	-0.000562 * color**3 + 0.044390 * color**2 + 0.355123 * color + 1.043270
            bBP	=	-0.000400 * color**3 + 0.018878 * color**2 + 0.195768 * color + 1.465592
            cBP	=	+0.000262 * color**3 + 0.060769 * color**2 - 0.205807 * color - 1.866968
            
            # WARNING !!!! this needs to be updated to correct formulas!!!
            z=[]
            for i,mag in enumerate(isoc['Gmag']):
                z.append(np.max([10**(0.4*(11 - 15)), 10**(0.4*(mag - 15.))]))
            er = 1.e-3*np.sqrt(10**aBP*np.array(z)**2 + 10**bBP*np.array(z) + 10**cBP)
            isoc = add_col(isoc,er,'e_G_BPmag')
            
        elif (filter == 'G_RPmag'):
            color = isoc['G_BPmag'] - isoc['G_RPmag']
            aRP	=	-0.007597 * color**3 + 0.114126 * color**2 - 0.636628 * color + 1.615927
            bRP	=	-0.003803 * color**3 + 0.057112 * color**2 - 0.318499 * color + 1.783906
            cRP	=	-0.001923 * color**3 + 0.027352 * color**2 - 0.091569 * color - 3.042268

            # WARNING !!!! this needs to be updated to correct formulas!!!
            z=[]
            for mag in isoc['Gmag']:
                z.append(np.max([10**(0.4*(11 - 15)), 10**(0.4*(mag - 15.))]))
            er = 1.e-3*np.sqrt(10**aRP*np.array(z)**2 + 10**bRP*np.array(z) + 10**cRP)
            isoc = add_col(isoc,er,'e_G_RPmag')
            
        else:
            er = er_coefs[filter][0] +  er_coefs[filter][1] * np.exp(isoc[filter]/er_coefs[filter][2])
            er[er > 0.5*3.] = 0.5*3.        
    
        
    return isoc


###############################################
# king profile: y = Nbg +(Nc/(1+(R/Rcore)^2)) 
# where rcore is the cluster core radius, Ncore is the core density,
# and Nbg the background density level, respectively. The core radius 
# was defined as a distance where density drops to half of Ncore. 
    
def king_prof(radius, rcore, Ncore, Nbg):
    '''
    Three parameters King profile fit.
    '''
    return Nbg + Ncore / (1.0 + (radius/rcore)**2)


###############################################
# Sample from King profile to get radius vector
    
def sample_king(nstars, rcore, Ncore, mass):
    
    rad_int = np.arange(0.,2.5*rcore,2.5*rcore/100.)
    king_val = king_prof(rad_int, rcore, Ncore, 0.)
    
    #normalize
    king_norm =  king_val / trapezoid(king_val,rad_int)
    
    # get cumulative distribution
    cum_king = cumulative_trapezoid(king_norm,rad_int, initial=0)
    
    # sample from King profile
    gen_radius = ( (interp1d(cum_king,rad_int))(np.random.rand(nstars)) )
            
    plt.plot(mass,gen_radius,'o')

    return np.array(gen_radius)

###############################################
# Sample from King profile to get radius vector
    
def sample_king_test(nstars, rcore, Ncore, mass):
    
    # define 4 zones to sample radius
    rout = 2.*rcore
    rad_int_1 = np.arange(0.,rout*0.5,rout*0.5/100.)
    rad_int_2 = np.arange(0.,rout*0.8,rout*0.8/100.)
    rad_int_3 = np.arange(0.,rout,rout/100.)
    rad_int_4 = np.arange(0.,rout,rout/100.)
        
    Mmin, Mmax = np.min(mass), np.max(mass)
    
    gen_radius=[]
    
    dm = (Mmax-Mmin)/4.
    for m in mass:
        
        if (m>=Mmin and m<Mmin+dm):
            king_val = king_prof(rad_int_4, rcore, Ncore, 0.)
            #normalize
            king_norm =  king_val / trapezoid(king_val,rad_int_4)
            # get cumulative distribution
            cum_king = cumulative_trapezoid(king_norm,rad_int_4, initial=0)
            # sample from King profile
            gen_radius.append( (interp1d(cum_king,rad_int_4))(np.random.rand()) )
            
        if (m>=Mmin+dm and m<Mmin+2*dm):
            king_val = king_prof(rad_int_3, rcore, Ncore, 0.)
            #normalize
            king_norm =  king_val / trapezoid(king_val,rad_int_3)
            # get cumulative distribution
            cum_king = cumulative_trapezoid(king_norm,rad_int_3, initial=0)
            # sample from King profile
            gen_radius.append( (interp1d(cum_king,rad_int_3))(np.random.rand()) )
            
        if (m>=Mmin+2*dm and m<Mmin+3*dm):
            king_val = king_prof(rad_int_2, rcore, Ncore, 0.)
            #normalize
            king_norm =  king_val / trapezoid(king_val,rad_int_2)
            # get cumulative distribution
            cum_king = cumulative_trapezoid(king_norm,rad_int_2, initial=0)
            # sample from King profile
            gen_radius.append( (interp1d(cum_king,rad_int_2))(np.random.rand()) )
            
        if (m>=Mmin+3*dm and m<Mmin+4*dm or m==Mmax):
            king_val = king_prof(rad_int_1, rcore, Ncore, 0.)
            #normalize
            king_norm =  king_val / trapezoid(king_val,rad_int_1)
            # get cumulative distribution
            cum_king = cumulative_trapezoid(king_norm,rad_int_1, initial=0)
            # sample from King profile
            aux = (interp1d(cum_king,rad_int_1))(np.random.rand()) 
            gen_radius.append( aux )

    return np.array(gen_radius)

##############################################################################################
# convert radius vector in arcmin coordinate pair 
# in decimal degree assming angular symetry
    
def gen_cluster_coordinates(ra0,dec0,nstars, rcore, Ncore,mass):
    
    # sample radius from King profile
    radius = sample_king_test(nstars, rcore, Ncore, mass)*u.arcmin
    
    #convert arcmin to degree
    radius = radius.to(u.deg)
    
    
    # get uniform distribution of angle [0,2Pi]
    phi = np.random.uniform(0,2*np.pi,nstars)
    
    # convert (r,phi) to (ra,dec)
    ra = radius*np.cos(phi) + ra0*u.deg
    dec = radius*np.sin(phi) + dec0*u.deg
    
    return ra.value,dec.value


###############################################
# generate a sample of field star coordinates
# assuming uniform distribution and given a field size
    # in arcmin
    
def gen_field_coordinates(ra0,dec0,field_sz, nstars):
    
    field_sz = (field_sz*u.arcmin).to(u.deg)
    ra0, dec0 = ra0*u.deg, dec0*u.deg

    ra = np.random.uniform((ra0-field_sz/2.).value,(ra0+field_sz/2.).value,nstars)
    dec = np.random.uniform((dec0-field_sz/2.).value,(dec0+field_sz/2.).value,nstars)

    return ra,dec


###############################################
# generate a synthetic cluster
    
def model_cluster(age,dist,FeH,Av,bin_frac,nstars,bands,refMag,Mcut=False,error=True,
                  seed=None,Abscut=False,imf='chabrier',alpha=2.3,beta=-3,gaia_ext = True):
    
    # check if binarity is ok
    if( bin_frac < 0. or bin_frac > 1.):
        #print 'binarity out of intervl, setting it to 1...'
        bin_frac = 1.
    
    # check if distance is ok    
    if( dist < 0.01 ):
        dist = 0.01
        
    # check age is ok    
    if( age < 6.65):
        age = 6.65
    elif (age>10.3):
        age = 10.3
    
    # check metalicity is ok    
    if( FeH < -0.9):
        FeH = -0.9
    elif (FeH>0.7):
        FeH = 0.7
    
    # check Av is ok    
    if( Av < 0.01):
        Av = 0.01
    elif (Av>6.0):
        Av = 6.0
    
    
    # get isochrone
    met = (10.**FeH)*0.0152
    grid_iso = get_iso_from_grid(age,met,bands,refMag,Abscut=Abscut)

    # make an observed isochrone
    obs_iso = make_obs_iso(bands, grid_iso, dist, Av, gaia_ext = gaia_ext)

#    use_imf = 'salpeter' 
#    use_imf = 'chabrier' 
    #use_imf = 'MillerScalo' 
#    use_imf = 'deMarchi'

    #sample from isochrone
    gen_iso = sample_from_isoc(obs_iso,bands,refMag,nstars,imf=imf,alpha=alpha, 
                               beta=beta, Mcut=Mcut,seed=seed)
    gen_iso_bin = sample_from_isoc(obs_iso,bands,refMag,nstars,imf=imf,alpha=alpha, 
                               beta=beta,Mcut=False,seed=seed)

    # add some binaries
    gen_iso = add_binaries(bin_frac,gen_iso,gen_iso_bin,bands,refMag,imf=imf,
                           alpha=alpha,beta=beta,seed=seed,binratio=0.8)

    # add some errors
    if error:
        gen_iso = add_phot_errors(gen_iso,bands)
        gen_iso_er = get_phot_errors(gen_iso,bands)


    return gen_iso


##############################################################################
# Define the log likelihood
# theta -> vector of model parameters [age, dist, z, ebv, Rv]
# 

def lnlikelihoodCE_log(theta, obs_iso, obs_iso_er, bands, refMag, prange, weight, prior=np.array([[1.],[1.e3]]), seed=None):
    # LOG OPTIMIZED (Numpy)
    age, dist, FeH, Av = theta
    bin_frac = 0.5
    nstars = obs_iso.size
    Mlim = obs_iso[refMag].max()
    
    # --- OPTIMIZER SAFETY NET ---
    try:
        # get synth isochrone
        mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,2000,bands,
                                    refMag,error=False,Mcut=Mlim,seed=seed,
                                    imf='chabrier',alpha=2.1, beta=-3.,gaia_ext=True)
        
        # Also reject if the model successfully runs but returns no stars
        if len(mod_cluster) == 0:
            return 1e10
            
    except ValueError:
        # Catch the argmin crash from sample_from_isoc and penalize the CE algorithm
        return 1e10 
    # ----------------------------
   
    obs = np.column_stack([obs_iso[b] for b in bands])
    obs_er = np.column_stack([obs_iso_er[b] for b in bands])
    mod = np.column_stack([mod_cluster[b] for b in bands])
    
    norm = np.prod(1. / np.sqrt(2.*np.pi * obs_er**2), axis=1) # (N,)
    inv_sigma2 = 1.0 / (obs_er**2)
    
    p_iso = np.empty(nstars)
    
    for i in range(nstars):
        diff = obs[i,:] - mod
        term_sum = np.sum(-0.5 * (diff**2) * inv_sigma2[i,:], axis=1)
        p_iso[i] = norm[i] * np.max(np.exp(term_sum))

    prior_term = np.prod(np.exp(-0.5*( (theta-prior[0,:])/prior[1,:] )**2))
    p_iso = p_iso * prior_term
    p_iso[p_iso < 1.e-307] = 1.e-307   
    res = np.log(p_iso) + np.log(weight)
    res = -np.sum(res)
    return res



def lnlikelihoodCE(theta,obs_iso,obs_iso_er,bands,refMag,prange,weight,
                   prior=[[1.],[1.e3]],seed=None):
    
    age, dist, FeH, Av = theta
    bin_frac = 0.5

    nstars = len(obs_iso) 
    Mlim = obs_iso[refMag].max()
    
    # --- OPTIMIZER SAFETY NET ---
    try:
        # get synth isochrone
        mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,2000,bands,
                                    refMag,error=False,Mcut=Mlim,seed=seed,
                                    imf='chabrier',alpha=2.1, beta=-3.,gaia_ext=True)
        
        # Also reject if the model successfully runs but returns no stars
        if len(mod_cluster) == 0:
            return 1e30
            
    except ValueError:
        # Catch the argmin crash from sample_from_isoc and penalize the CE algorithm
        return 1e30 
    # ----------------------------

    # get distance of each observed star to the model isochrone
    obs = np.array(obs_iso[bands].tolist())
    obs_er = np.array(obs_iso_er[bands].tolist())
    mod = np.array(mod_cluster[bands].tolist())
    
    p_iso = []
    for i in range(nstars):
        aux = np.prod(1./np.sqrt(2.*np.pi*obs_er[i,:]**2)*
                      np.exp(-0.5*(obs[i,:]-mod)**2/obs_er[i,:]**2),axis=1)
        p_iso.append(np.max(aux))

    p_iso = np.array(p_iso) 
    p_iso[p_iso < 1.e-307] = 1.e-307   
    
    res = np.log(p_iso) + np.log(weight)
    log_likelihood = np.sum(res)
    
    # Calculate log_prior mathematically correctly
    prior_arr = np.array(prior)
    log_prior = np.sum(-0.5 * ((theta - prior_arr[0,:]) / prior_arr[1,:])**2)
    
    total_log_posterior = log_likelihood + log_prior
    penalty = -total_log_posterior
    
    # Final check to prevent NaNs from crashing the CE optimizer
    if np.isnan(penalty) or penalty > 1e30:
        return 1e30
        
    return penalty


def lnlikelihoodCE_ori(theta,obs_iso,obs_iso_er,bands,refMag,prange,weight,
                   prior=[[1.],[1.e3]],seed=None):
    
    # generate synth cluster from input parameters
    age, dist, FeH, Av = theta
    
    bin_frac = 0.5

    # number of stars to generate
    nstars = obs_iso.size
    Mlim = obs_iso[refMag].max()
    
    # --- OPTIMIZER SAFETY NET ---
    try:
        # get synth isochrone
        mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,2000,bands,
                                    refMag,error=False,Mcut=Mlim,seed=seed,
                                    imf='chabrier',alpha=2.1, beta=-3.,gaia_ext=True)
        
        # Also reject if the model successfully runs but returns no stars
        if len(mod_cluster) == 0:
            return 1e30
            
    except ValueError:
        # Catch the argmin crash from sample_from_isoc and penalize the CE algorithm
        return 1e30 
    # ----------------------------
    
    # get distance of each observed star to the model isochrone
    obs = np.array(obs_iso[bands].tolist())
    obs_er = np.array(obs_iso_er[bands].tolist())

    mod = np.array(mod_cluster[bands].tolist())
    
    
    p_iso = []
    for i in range(nstars):
        
        aux = np.prod(1./np.sqrt(2.*np.pi*obs_er[i,:]**2)*
                      np.exp(-0.5*(obs[i,:]-mod)**2/obs_er[i,:]**2),axis=1)
        
        p_iso.append(np.max(aux))

    p_iso = np.array(p_iso)*np.prod(np.exp(-0.5*( (theta-prior[0,:])/prior[1,:] )**2))
        
    p_iso[p_iso< 1.e-307] = 1.e-307   
    res = np.log(p_iso) + np.log(weight)
    res = -np.sum(res)
    
    #print ('1:',res, '2:',p_iso.max(), '3:',p_iso.min()) #'   '.join('%0.3f' % v for v in theta)
        
    return res


##############################################################################
# Define the log likelihood
# theta -> vector of model parameters [age, dist, z, ebv, Rv]
# 
def lnlikelihood(theta,obs_iso,obs_iso_er,bands,refMag,prange,seed=None):
    
    # generate synth cluster from input parameters
    age, dist, FeH, Av = theta
    
    bin_frac = 0.5

    # number of stars to generate
    nstars = obs_iso.size
    Mlim = obs_iso[refMag].max()
    
    # get synth isochrone

    mod_cluster = model_cluster(age,dist,FeH,Av,bin_frac,2000,bands,
                                refMag,error=False,Mcut=Mlim,seed=seed,
                                imf='chabrier',alpha=2.1, beta=-3.,gaia_ext = True)

    # get distance of each observed star to the model isochrone
    obs = np.array(obs_iso[bands].tolist())
    obs_er = np.array(obs_iso_er[bands].tolist())

    mod = np.array(mod_cluster[bands].tolist())

    p_iso = []
    for i in range(nstars):
        
        aux = np.prod(1./np.sqrt(2.*np.pi*obs_er[i,:]**2)*
                      np.exp(-0.5*(obs[i,:]-mod)**2/obs_er[i,:]**2),axis=1)
        
        p_iso.append(np.max(aux))

    p_iso = np.array(p_iso)        
    p_iso[p_iso< 1.e-307] = 1.e-307   
        
    return np.log(p_iso)


##############################################################################
# Define the log posterior
# theta -> vector of model parameters [age, dist, z, ebv, Rv]
# 
def lnposterior(theta,loglikelihood,weight,prior=[[1.],[1.e3]]):
    
    age, dist, FeH, Av = theta
    
    res = np.log(p_iso) + np.log(weight) + np.log(np.prod(np.exp(-0.5*( (theta-prior[0,:])/prior[1,:] )**2)))
    
