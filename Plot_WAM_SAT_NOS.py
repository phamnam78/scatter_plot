#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:56:13 2023

@author: Nam Thanh Pham, Hemholtz-Zentrum Hereon, Geesthacht, Germany

scatter and qq plot of model and satellite data

"""

import numpy as np
import netCDF4
import glob
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size':14})

def density_scatter( x , y, fig = None, ax = None, sort = True, bins = 20, DAT='', **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    # x_min=0.;x_max=3; dx=(x_max-x_min)/20
    # x_min=int(y.min());x_max=int(y.max())+2; dx=(x_max-x_min)/20
    x_min=0;x_max=12; dx=(x_max-x_min)/20
    # print(x_min,x_max)
    
    if ax is None :
        fig , ax = plt.subplots()

    fig.set_figheight(6)
    fig.set_figwidth(6.8)
    
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
    norm = Normalize(vmin = 0, vmax = np.max(z))
    ax.scatter( x, y, c=z, s=5, norm=norm, **kwargs )
    
    cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('Density')

    ax.set_xlabel('SAT DATA, Hs (m)',fontsize=14,fontweight='bold')
    ax.set_ylabel('MODEL DATA, Hs (m)',fontsize=14,fontweight='bold') #('GCOAST Data')
    ax.set_title(DAT,fontsize=16,fontweight='bold')
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(x_min,x_max)
    xticks=np.linspace(x_min,x_max,7)
    yticks=np.linspace(x_min,x_max,7)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    xx = np.linspace(x_min,x_max)
    ax.plot(xx,xx, color="b", ls="-")

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    yy=slope*xx
    ax.plot(xx,yy,color="y",ls='-')
       
    ax.text(x_max-9.5*dx,x_min+0.9*dx,'y = '+ "{:.2f}".format(slope)+' x ',fontsize=14)

    percs = np.linspace(0,100,100)
    qn_x = np.percentile(x, percs)
    qn_y = np.percentile(y, percs)

    ax.plot(qn_x,qn_y, ls="", marker="+",color='red')

    mean_R,mean_M,std_R,std_M,bias,n_bias,rmse,si,hh,skill_score,corr=stats_index(x,y)
    
    ax.text(x_min+0.07,x_max-dx,'Mean R = '+ "{:.2f}".format(mean_R)+' m',fontsize=14)
    ax.text(x_min+6,x_max-dx,'Std R = '+ "{:.2f}".format(std_R)+' m',fontsize=14)
    ax.text(x_min+0.07,x_max-2*dx,'mean M = '+ "{:.2f}".format(mean_M)+' m',fontsize=14)
    ax.text(x_min+6,x_max-2*dx,'Std M = '+ "{:.2f}".format(std_M)+' m',fontsize=14)
    ax.text(x_min+0.07,x_max-3*dx,'bias = '+ "{:.2f}".format(bias*100)+' cm',fontsize=14)
    ax.text(x_min+0.07,x_max-4*dx,'rmse = '+ "{:.2f}".format(rmse*100)+' cm',fontsize=14)
    ax.text(x_min+0.07,x_max-5*dx,'si = '+ "{:.2f}".format(si),fontsize=14)
    ax.text(x_min+0.07,x_max-6*dx,'hh = '+ "{:.2f}".format(hh),fontsize=14)
    ax.text(x_min+0.07,x_max-7*dx,'skill = '+ "{:.2f}".format(skill_score),fontsize=14)
    ax.text(x_min+0.07,x_max-8*dx,'corr = '+ "{:.2f}".format(corr),fontsize=14)

    return ax


#####################################################################################################
def stats_index(x,y):
    
    errors=y-x

    mean_R=np.mean(x)
    mean_M=np.mean(y)
    std_R=np.std(x)
    std_M=np.std(y)
    print('Mean R: {:.2f}'.format(mean_R),'Mean M: {:.2f}'.format(mean_M))
    print('Std R: {:.2f}'.format(std_R),'Std M: {:.2f}'.format(std_M))
     
    #### bias
    bias=mean_M - mean_R
    #### normalized bias
    n_bias = bias/mean_R
    print('Norm. Bias: {:.2f}'.format(n_bias))
    ### root-mean-square error
    rmse=np.sqrt(np.mean(errors*errors))
    ### scatter index
    si = rmse/mean_R
    print('si index:{:.2f}'.format(si))
    #### HH index
    hh = np.sqrt(np.sum(errors*errors)/np.sum(x*y))
    print ('hh index:{:.2f}'.format(hh))
    #### Willmott et al. (1985) index
    skill_score = 1 - np.sum(np.abs(errors))/np.sum(np.abs(y-mean_R)+np.abs(x-mean_R))
    print ('Skill score:{:.2f}'.format(skill_score))

    tmp_corr=np.corrcoef(x,y)
    corr=tmp_corr[0,1]
    print ('Corr Coef:{:.2f}'.format(corr))
    
    return mean_R,mean_M,std_R,std_M,bias,n_bias,rmse,si,hh,skill_score,corr 

####################################################################################

#####################################################################################################

path_file= '/work/gg0028/g260162/FOR_RUMEYSA/PAIR_DATA/'

dir_out= '/work/gg0028/g260162/TMP/PLOT_QQ/'

name_alt=[]
# name_alt=np.append(name_alt,'Al')
name_alt=np.append(name_alt,'C2')
# name_alt=np.append(name_alt,'CFO')
# name_alt=np.append(name_alt,'H2b')
# name_alt=np.append(name_alt,'J3')
# name_alt=np.append(name_alt,'S3a')
# name_alt=np.append(name_alt,'S3b')

# name_alt=np.append(name_alt,'ALL SAT')

num_alt=len(name_alt)

EXP_name='AMM15'

# EXP_name='ORG_AMM15'

ALL_MOD=[]; ALL_SAT=[]
for i in range(num_alt):

    file_pair='{}{}_{}.nc'.format(path_file,EXP_name,name_alt[i])
    
    
    f1=netCDF4.Dataset(file_pair)

    WAM_data=np.array(f1.variables['mod_hs'][:])
    SAT_data=np.array(f1.variables['alt_hs'][:])
    
    diff = SAT_data-WAM_data
    idx = np.where(np.abs(diff) < 2)[0]
    nn = idx.shape[0]
    new_WAM = np.zeros((nn))
    new_SAT = np.zeros((nn))

    for k in range (nn):
        new_WAM[k]=WAM_data[idx[k]]
        new_SAT[k]=SAT_data[idx[k]]
    
    ALL_MOD=np.append(ALL_MOD,new_WAM)
    ALL_SAT=np.append(ALL_SAT,new_SAT)
        
        
    
print(ALL_MOD.shape[0])
###### plot ALL 
fig, ax = plt.subplots() 

tmp = density_scatter(x = ALL_SAT, y = ALL_MOD, fig = fig, ax=ax, bins = [30,30], DAT='QQ PLOT OF NWS_AMM15 vs SAT')
file_out='{}{}_ALL_SAT.png'.format(dir_out,EXP_name)
# plt.savefig(file_out,dpi=300,bbox_inches='tight')


