#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

esse programa ajusta isocronas aos dados gaia DR3
busca todos os arquivos .csv gerados pelo HDBSCAN (ja formatados) de um diretorio
ajusta a iso em cada um deles
escreve os todos os parametros (cinematicos + astrofisicos) num log e tambem no diretorio de cada arquivo


== voce deve inserir o arquivo log-results.txt (modelo) dentro do /results


@author: Hektor + wilton em 29 junho 2021

wilton = 08dez25

"""
import os
# to disable numpy multithreading
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import mkl
mkl.set_num_threads(1)
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import astropy.units as u
import warnings
import astropy.coordinates as coord
from oc_tools_padova_edr3 import *
from gaia_edr3_tools import *
import glob
import os
import statistics
import shutil

warnings.filterwarnings("ignore")
##########################################################################################
#vc adiciona col, com o nome col_name no array
#col_name = voce definie = string  = nome da col
#array = arquivo lido
#col = a coluna de valores que vc quer adicionar = valor
#Ex:teste = add_col(catresmasers,RAICRS,'RA_ICRS') 
def add_col(array,col,col_name):
    col_type=(col_name, col.dtype)
    y=np.zeros(array.shape, dtype=array.dtype.descr+[col_type])    
    for name in array.dtype.names: y[name] = array[name]    
    y[col_type[0]]=col    
    return y
##########################################################################################
#array = obs['ag_gspphot']  == equivale ao array com os dados
#Gmag = transforma_para_float(obs['phot_bp_mean_mag']) = exemplo de como usar
def transforma_para_float(array):
    # para transformar dados em formato U9 para float
    saida = array.astype(str)
    n_elementos = len(array)
    for j in range(n_elementos):
        if saida[j] == 'nan4':
            saida[j] = 'nan'
            saida[j] = ''
        else:
            saida[j] = float(saida[j])
    saida = b = np.asarray(saida,dtype = float)
    return saida
##########################################################################################
plt.close('all')
############################################################################
# setar diretorios
base_dir = '/home/hmonteiro/Google Drive/work/clusters/gaia_dr3/CE_IsoFit/'
#fit_dir = base_dir+'OCFit/'
memb_dir = base_dir+'dados/'
dirout = base_dir+'results/' 
logfilename = 'log-results.txt'
############################################################################
#cria diretorio results
try:
    os.stat(dirout)
except:
    os.mkdir(dirout)  
###########################################################################
# Get Apogee data from disk
apogee = np.load(base_dir+'APOGEE-Metalicity-table.npy')
############################################################################
# Get Netopil data from disk
netopil = np.load(base_dir+'Netopil16-Metalicity-table.npy')
#netopil['Cluster'] = np.array([x.decode().replace(" ", "_") for x in netopil['Cluster']])
netopil['Cluster'] = np.array([x.replace(" ", "_") for x in netopil['Cluster']])
############################################################################
# verifica se ja foi rodado
catlist = np.genfromtxt(dirout+'log-results.txt',dtype=None,delimiter=';',names=True)
catlist['name'] = np.array([x.strip() for x in catlist['name']])
good_OCs = catlist['name'].astype(str) # para escrever os nomes em str
############################################################################
# serve apenas para escrever o header. nao e necessario quando usado o arquivo log-results.txt no dir results
#logfile = open(dirout+logfilename, "a+")
#logfile.write('name;                    RA_ICRS ;   DE_ICRS;        R;     dist;  e_dist;     age;   e_age;     FeH;   e_FeH;      Av;    e_Av;      Nc       \n')
#logfile.close
############################################################################
# set membership files that will be fit sorted by size
files = [f for f in sorted(glob.glob(memb_dir + "*.dat", recursive=True),key=os.path.getsize)]

contfiles = 0

for i in range(len(files)):
    
    plt.close('all')
    
    if '__pycache__' in base_dir:
           shutil.rmtree(base_dir+'__pycache__')
           shutil.rmtree(base_dir+'joblib')    
    
    logfile = open(dirout+'/'+logfilename, "a+")

    filename_w_ext = os.path.basename(files[i])
    name, file_extension = os.path.splitext(filename_w_ext)
    
    # fit only good OCs
    if (name in good_OCs):
        print('cluster ',name,' Ja foi rodado.')
        continue
    
    print('####################################################################')
    print('Runing cluster:',name, '   ', 'number = ',i)
    
    # create directory for results of the cluster
    try:
        os.stat(dirout+name)
    except:
        os.mkdir(dirout+name)       
    
    # file to save all output
    verbosefile = open(dirout+name+'/'+'verbose-output.dat', 'w') 
    
    magcut = 40.
    guess = False
   
    obs = np.genfromtxt(files[i],names=True, delimiter=';', dtype=None, missing_values='',filling_values='nan',encoding='utf-8')
    
    #stop
    #remove nans para fazer os plots
    Gmag = transforma_para_float(obs['phot_g_mean_mag'])
    BPmag = transforma_para_float(obs['phot_bp_mean_mag'])
    RPmag = transforma_para_float(obs['phot_rp_mean_mag'])
    
    cond1 = np.isfinite(Gmag)
    cond2 = np.isfinite(BPmag)
    cond3 = np.isfinite(RPmag)
    ind  = np.where(cond1&cond2&cond3)
    
    obs = obs[ind]
    RA = obs['ra']
    DEC = obs['dec']
    Plx = transforma_para_float(obs['parallax'])
    ePlx = transforma_para_float(obs['parallax_error'])
    pmra = transforma_para_float(obs['pmra'])
    pmde = transforma_para_float(obs['pmdec'])
    RV =  transforma_para_float(obs['radial_velocity'])
    erRV =  transforma_para_float(obs['radial_velocity_error'])
    Gmag = transforma_para_float(obs['phot_g_mean_mag'])
    BPmag = transforma_para_float(obs['phot_bp_mean_mag'])
    RPmag = transforma_para_float(obs['phot_rp_mean_mag'])
    Ag = transforma_para_float(obs['ag_gspphot'])
    members = transforma_para_float(obs['probs'])
    weight = transforma_para_float(obs['probs'])
    n_members = len(RA)
    
    ###########################################################################
    # calculo dos erros e G, B, R
    #They are obtained with a simple propagation of errors with the formulas
    #with the G, G_BP, G_RP zero point uncertainties
    sigmaG_0 = 0.0027553202
    sigmaGBP_0 = 0.0027901700
    sigmaGRP_0 = 0.0037793818
    phot_g_mean_flux_error = transforma_para_float(obs['phot_g_mean_flux_error'])
    phot_g_mean_flux = transforma_para_float(obs['phot_g_mean_flux'])
    phot_bp_mean_flux_error = transforma_para_float(obs['phot_bp_mean_flux_error'])
    phot_bp_mean_flux = transforma_para_float(obs['phot_bp_mean_flux'])
    phot_rp_mean_flux_error = transforma_para_float(obs['phot_rp_mean_flux_error'])
    phot_rp_mean_flux = transforma_para_float(obs['phot_rp_mean_flux'])
    
    eGmag = np.sqrt((-2.5/np.log(10)*phot_g_mean_flux_error/phot_g_mean_flux)**2 + sigmaG_0**2)
    eBPmag = np.sqrt((-2.5/np.log(10)*phot_bp_mean_flux_error/phot_bp_mean_flux)**2 + sigmaGBP_0**2)
    eRPmag = np.sqrt((-2.5/np.log(10)*phot_rp_mean_flux_error/phot_rp_mean_flux)**2 + sigmaGBP_0**2)
    ###########################################################################
    # vou acrescentar as colunas no obs para nao modificr as rotinas usadas no iso-fit
    obs = add_col(obs,Gmag,'Gmag')
    obs = add_col(obs,eGmag,'e_Gmag')
    obs = add_col(obs,BPmag,'BPmag')
    obs = add_col(obs,eBPmag,'e_BPmag')    
    obs = add_col(obs,RPmag,'RPmag')
    obs = add_col(obs,eRPmag,'e_RPmag')
    ###########################################################################
    # Calculate cluster radius
    # calculate center coordinate
    center = coord.SkyCoord(ra=np.mean(RA)*u.degree, dec=np.mean(DEC)*u.degree)
    
    # coordinates of stars
    star_coords = coord.SkyCoord(ra=RA*u.degree, dec=DEC*u.degree)
    
    # obtain distance from center
    dist_center = center.separation(star_coords).degree  
    
    # adopted radius of the cluster
    radius = np.max(dist_center)
    #######################################################################
    member_cut = 0.51
    
    if (members.max() <= member_cut):
        print(name,' cluster has no stars with P > ', member_cut)
        continue
    ind_m = members > member_cut
    
    Nmembers = len(ind_m)
   
    filters = ['Gmag','G_BPmag','G_RPmag']
    refmag = 'G_BPmag'
    ####################################################################################
    # priors
    
    guess_dist = infer_dist(Plx[ind_m], ePlx[ind_m],guess=1./Plx[ind_m].mean())
    print('Infered distance from parallax: %8.3f \n'%(guess_dist))
    #    verbosefile.write('Infered distance from parallax: %8.3f \n'%(guess_dist))
    dist_posterior_x=[]
    dist_posterior_y=[]
    for d in np.linspace(0.01,3*guess_dist,1000): 
        dist_posterior_x.append(d)
        dist_posterior_y.append(-likelihood_dist(d,Plx[ind_m], ePlx[ind_m]))
    dist_posterior_x = np.array(dist_posterior_x)
    dist_posterior_y = np.array(dist_posterior_y)
    dist_posterior_y[dist_posterior_y<0.]=0
    cum = np.cumsum(dist_posterior_y)/np.sum(dist_posterior_y)
    #    conf_int = np.where((cum > 0.16)&(cum<0.84))[0]
    conf_int = np.where((cum > 0.16)&(cum<0.5))[0]
    try:
    #        dist_guess_sig = (dist_posterior_x[conf_int[-1]] - dist_posterior_x[conf_int[0]])/2.
        dist_guess_sig = (dist_posterior_x[conf_int[-1]] - dist_posterior_x[conf_int[0]])
    
    except:
        print('using rough distance interval estimate...')
        avg_plx = np.sum(Plx[ind_m]/ePlx[ind_m])/np.sum(1./ePlx[ind_m])
        er_plx = np.mean(ePlx[ind_m])
        if (avg_plx > er_plx):
            dist_guess_sig = ( 1./(avg_plx-1*er_plx) - 
                              1./(avg_plx+1*er_plx) )/2.
        else:
            dist_guess_sig = np.min([0.5*guess_dist,1.])

    if (guess_dist <= 0.0):
        condA1 =Plx>0.0
        condA2 = members > member_cut
        indplxpositive = np.where(condA1&condA2)
        guess_dist = 1./Plx[indplxpositive].mean()
        dist_guess_sig = 1./Plx[indplxpositive].std()

    print ('Mag. Cut:')
    print(magcut)

    verbosefile.write('Mag. Cut: \n')
    verbosefile.write(str(magcut)+'\n')
         
    print ('Membership Cut:')
    print(member_cut)

    verbosefile.write('Membership Cut: \n')
    verbosefile.write(str(member_cut)+'\n')
         
    ###########################################################################  
    # para guess Av: usando Ag do catalogo GAIA DR3 usando AG ≈ 0.859⋅AV
    #remove nans para fazer os plots
    cond1_Ag = np.isfinite(Ag)
    cond_member = weight > 0.50
    
    ind_Ag = np.where(cond1_Ag&cond_member)
    
    Agmean = np.mean(Ag[ind_Ag])
    Agstd = np.std(Ag[ind_Ag])
    NAg = len(ind_Ag[0])
    Av_guess = Agmean/0.859
    Av_guess_sig = 0.3*Av_guess
    
    print('From Gaia AG ≈ 0.859⋅AV: Av = {:.3f} +/- {:.3f} mag'.format(Av_guess,Av_guess_sig))
    verbosefile.write('From Gaia AG ≈ 0.859⋅AV: Av =  {:.3f} +/- {:.3f} mag \n'.format(Av_guess,Av_guess_sig))
    
    if (~np.isfinite(Av_guess)):
        print('no data Ag from Gaia')
        verbosefile.write('no data Ag from Gaia...')
        # vou usar a estimativa 1 magnitudes/kpc
        Av_guess = 1*guess_dist
        Av_guess_sig = 1.0e3
    
    ###########################################################################
    # check to see if there is a metalicity determinantion to use in prior
    # priority is for Netopil HighRes
    if (name.encode() in netopil['Cluster']):
        inda = np.where(netopil['Cluster'] == name.encode())[0][0]
        if (np.isfinite(netopil['__Fe_H_HQS'][inda])):
            guess_FeH =netopil['__Fe_H_HQS'][inda]
            if(np.isfinite(netopil['e__Fe_H_HQS'][inda]) and netopil['e__Fe_H_HQS'][inda] > 0.):
                guess_FeH_sig = netopil['e__Fe_H_HQS'][inda]
            else:
                guess_FeH_sig = 0.1
            print('Using Netopil HQS FeH value in prior...')
            verbosefile.write('Using Netopil HQS FeH value in prior...\n')
        elif(np.isfinite(netopil['__Fe_H_LQS'][inda])):
            guess_FeH =netopil['__Fe_H_LQS'][inda]
            if(np.isfinite(netopil['e__Fe_H_LQS'][inda]) and netopil['e__Fe_H_LQS'][inda] > 0.):
                guess_FeH_sig = netopil['e__Fe_H_LQS'][inda]
            else:
                guess_FeH_sig = 0.1
        else:
            guess_FeH = 0.0
            guess_FeH_sig = 0.1
            print('Using Netopil LQS FeH value in prior...')
            verbosefile.write('Using Netopil LQS FeH value in prior...\n')
            
    elif (name.encode() in apogee['Cluster']):
        inda = np.where(apogee['Cluster'] == name.encode())[0][0]
        guess_FeH =apogee['__Fe_H_'][inda]  
        guess_FeH_sig = apogee['e__Fe_H_'][inda]
        if (~np.isfinite(guess_FeH)):
            guess_FeH = 0.
            guess_FeH_sig = 1.e3
        print('Using Apogee FeH value in prior...')
        verbosefile.write('Using Apogee FeH value in prior...\n')
        
    else:
        # get galactic coordinates for apogee sample
        coords = coord.SkyCoord(RA.mean()*u.deg,DEC.mean()*u.deg, distance=guess_dist*u.kpc)
        c2 = coords.transform_to(coord.Galactocentric)    
        gal_coords = [c2.x.to(u.kpc),c2.y.to(u.kpc),c2.z.to(u.kpc)]    
        GCradius = np.sqrt(gal_coords[0]**2+gal_coords[1]**2).value
        
        # calculate FeH from gradient
        if(GCradius < 13.9): 
            guess_FeH = -0.068 * (GCradius-8.0)
            guess_FeH_sig = 0.1
        else:
            guess_FeH = -0.009 * (GCradius-13.9) - 0.4
            guess_FeH_sig = 0.1
      
        print('Galactocentric radius: ',GCradius)
        verbosefile.write('Galactocentric radius: %8.2f \n'%GCradius)
        print('FeH prior from Galactic Gradient: %8.2f +- %8.2f'%(guess_FeH,guess_FeH_sig))
        verbosefile.write('FeH prior from Galactic Gradient: %8.2f +- %8.2f \n'%(guess_FeH,guess_FeH_sig))

    guess = [8.0,guess_dist,guess_FeH,Av_guess]
    guess_sig = np.array([1.e3, dist_guess_sig, guess_FeH_sig, Av_guess_sig])      

    prior = np.stack([guess,guess_sig])
    
    print ('Guess:')
    print(guess)
    print('Prior sig: ')
    print(guess_sig)
    
    verbosefile.write('Guess: \n')
    verbosefile.write(str(guess)+'\n')
    
    verbosefile.write('Prior sigma: \n')
    verbosefile.write(str(guess_sig)+'\n')
    
    npoint = np.where(BPmag[ind_m] < magcut)
    print ('number of member stars:', npoint[0].size)
    verbosefile.write('number of member stars: %i \n'%npoint[0].size)
   
   # fit isochrone to the data
    res_isoc, res_isoc_er = np.array([]),np.array([])
    res_lik_median, res_lik_sigma = np.array([]),np.array([])
    res_lik_16th, res_lik_84th = np.array([]),np.array([])

    if (np.ravel(npoint).size > 5 and Gmag[ind_m].size < 10000):
    
        res_isoc, res_isoc_er, res_lik_median, res_lik_sigma, res_lik_16th, res_lik_84th  = fit_iso_GAIA(obs,verbosefile,
                                             guess, magcut, member_cut,
                                             obs_plx=Plx.mean(),
                                             obs_plx_er=Plx.std(), 
                                             prior=prior, 
                                             bootstrap=True,
                                             #fixFeH=guess_FeH+1.0e-6)
                                             fixFeH=False)
        ###########################################################################
        # parametros cinematicos do OC
        racenter_OC = statistics.median(RA[ind_m])
        decenter_OC = statistics.median(DEC[ind_m])
        Plx_OC = statistics.mean(Plx[ind_m])
        e_Plx_OC = statistics.stdev(Plx[ind_m])
        pmra_OC = statistics.mean(pmra[ind_m])
        e_pmra_OC = statistics.stdev(pmra[ind_m])
        pmde_OC = statistics.mean(pmde[ind_m])
        e_pmde_OC = statistics.stdev(pmde[ind_m])

        # para calcular R50 
        radius = pyasl.getAngDist(racenter_OC,decenter_OC,RA[ind_m],DEC[ind_m])
        radius = radius#*60. # in arcmin
        radius.sort()
        radiimax_OC = max(radius)
        radiimaxarcmin = radiimax_OC*60.      
        radius50_OC = np.sort(radius)[int(0.5*radius.size)]
        radius50a_OC = radius50_OC*60. # in arcmin
        # stars in r50
        condNr50_OC = radius<= radius50_OC
        ind_Nr50_OC = np.where(condNr50_OC)
        N_radius50_OC = len(ind_Nr50_OC[0])
        ##########################################################################################
        # Raio em pc usando r50
        R50pc_OC = np.ceil((radius50_OC*np.pi/180.)*res_isoc[1]*1.e3)
        ##########################################################################################
        # Raio em pc usando rmax
        Rmaxpc_OC = np.ceil((radiimax_OC*np.pi/180.)*res_isoc[1]*1.e3)
        ##########################################################################################
        # calcular velocidade radial sem outilers de 3sigma
        sig_clip = 3.
        cond1 = np.abs(RV[ind_m]-np.nanmean(RV[ind_m])) < sig_clip*np.nanstd(RV[ind_m])
        indRV  = np.where(cond1)
        RV_mean_OC = np.nanmean(RV[indRV])
        RV_std_OC = np.nanstd(RV[indRV])
        NstarsRV_OC = 0
        if (np.isfinite(RV_mean_OC) and np.isfinite(RV_std_OC)):
            indf = np.isfinite(RV[indRV])
            RV_mean_OC = np.sum((RV[indRV]/erRV[indRV])[indf])/np.sum((1./erRV[indRV])[indf])
            RV_std_OC = (RV[indRV])[indf].size/np.sum((1./erRV[indRV])[indf])
            NstarsRV_OC = np.count_nonzero(indf)
        ##########################################################################################
        # calcula coefciente de correlacao linear de Pearson cor com mag: r-value = res_r[0]
        Gmagv = Gmag[ind_m]
        BRmag = BPmag[ind_m]-RPmag[ind_m]
        res_r, N_sp = calcula_correlacao_ICxmag(Gmagv,BRmag)
        ##########################################################################################
        # to salve in a log file:
        logfile.write('{:<20s};'.format(name))                          
        logfile.write('%10.4f;'%racenter_OC)
        logfile.write('%10.4f;'%decenter_OC)
        logfile.write('%10.4f;'%radiimax_OC)
        logfile.write('%10.3f;'%pmra_OC)
        logfile.write('%10.3f;'%e_pmra_OC)
        logfile.write('%10.3f;'%pmde_OC)
        logfile.write('%10.3f;'%e_pmde_OC)
        logfile.write('%10.3f;'%Plx_OC)
        logfile.write('%10.3f;'%e_Plx_OC)
        logfile.write('%10.3f;'%RV_mean_OC)
        logfile.write('%10.3f;'%RV_std_OC)
        logfile.write('%8i;'%NstarsRV_OC)
        logfile.write('%10.4f;'%radius50_OC)
        logfile.write('%10.2f;'%R50pc_OC)
        logfile.write('%10.2f;'%Rmaxpc_OC)
        logfile.write('%8i;'%(res_isoc[1]*1.e3))                              
        logfile.write('%8i;'%(res_isoc_er[1]*1.e3))                              
        logfile.write('%8.3f;'%(res_isoc[0]))                              
        logfile.write('%8.3f;'%(res_isoc_er[0]))                              
        logfile.write('%8.3f;'%(res_isoc[2]))                              
        logfile.write('%8.3f;'%(res_isoc_er[2]))                              
        logfile.write('%8.3f;'%(res_isoc[3]))                              
        logfile.write('%8.3f;'%(res_isoc_er[3]))                              
        logfile.write('%8.3f;'%Agmean)                              
        logfile.write('%8.3f;'%Agstd) 
        logfile.write('%8i;'%NAg)              
        logfile.write('%8i;'%Nmembers) 
        logfile.write('%8i;'%N_sp) 
        logfile.write('%8i;'%N_radius50_OC)
        logfile.write('%12.2f;'%res_r[0])
        logfile.write('%12.3f;'%guess[0])
        logfile.write('%12.3f;'%guess_sig[0])
        logfile.write('%8.3f;'%guess[1])
        logfile.write('%8.3f;'%guess_sig[1])
        logfile.write('%8.3f;'%guess[2])
        logfile.write('%8.3f;'%guess_sig[2])
        logfile.write('%8.3f;'%guess[3])
        logfile.write('%8.3f;'%guess_sig[3])
        logfile.write('%12.3f;'%res_lik_median)
        logfile.write('%12.3f;'%res_lik_sigma)
        logfile.write('%12.3f;'%res_lik_16th)
        logfile.write('%12.3f'%res_lik_84th)
        logfile.write(' \n')
        logfile.close()
                              
        #########################################################################################
        # salva plot CMD
        grid_iso = get_iso_from_grid(res_isoc[0],(10.**res_isoc[2])*0.0152,filters,refmag, Abscut=False, nointerp=False)
        fit_iso = make_obs_iso(filters, grid_iso, res_isoc[1], res_isoc[3], gaia_ext = True)                
        # ordenar por membership para fazer o plot 
        ind = np.argsort(members)
        plt.figure(figsize=(5,6))
        ax1 = plt.subplot(1,1,1)
        plt.scatter(BPmag-RPmag,Gmag,s=2,color='lightgray',cmap='jet')
        plt.scatter(BPmag[ind_m]-RPmag[ind_m],Gmag[ind_m], cmap='jet', s=10,c=members[ind_m])
        plt.ylim(Gmag.max()+0.50,Gmag.min()-1.0)
        plt.xlim((BPmag-RPmag).min()-0.5,(BPmag-RPmag).max()+0.5)
        plt.plot(fit_iso['G_BPmag']-fit_iso['G_RPmag'],fit_iso['Gmag'], 'g',label='com polinomio',alpha=0.9)
        plt.xlabel('G_BP - G_RP [mag]',fontsize=15)
        plt.ylabel('G [mag]',fontsize=15)
        cbar=plt.colorbar()
        cbar.set_label('membership')
        ax1.tick_params(axis="both", labelsize=15, width=1.5)
        plt.title(name)
        plt.savefig(dirout+name+'/'+name+'.png', dpi=300)
        verbosefile.close()
        plt.close('all')
        #stop
        #########################################################################################
    

plt.close('all')  
print ('All done...')
    
    
    
