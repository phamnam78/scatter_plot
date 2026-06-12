#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:56:13 2023

@author: Nam Thanh Pham, Hemholtz-Zentrum Hereon, Geesthacht, Germany

pair data between the SAT and MODEL

"""
import numpy as np
import netCDF4
import xarray as xr
from datetime import datetime
import rioxarray

import warnings
warnings.filterwarnings("ignore")

#####################################################################################################
t1 = datetime.now()
st_time = t1.strftime("%H:%M:%S")

print("Start Time =", st_time)

# path_file='/scratch/g/g260162/AAA_LINK_FILES/GLO_WAVE/'
# path_file='/scratch/g/g260162/AAA_LINK_FILES/DWD_WAVE/'
path_file='/scratch/g/g260162/AAA_LINK_FILES/NWS_EIS/'

# path_file='/scratch/g/g260162/AAA_LINK_FILES/AMM15_WAVE/'

path_alt='/work/gg0028/g260162/VALIDATION/WAVE_DATA/FOR_NOS/'
dir_out='/work/gg0028/g260162/VALIDATION/WAVE_DATA/PAIR_DATA/'

all_BS=False  #True
# all_BS=True

list_sat=['Al','C2','CFO','H2b','H2c','J3','S3a','S3b','S6a']

list_mon=['01','02','03','04','05','06','07','08','09','10','11','12']

for sat_name in list_sat:
    print(sat_name)
    
    # file_out=dir_out+'GLO_{}.nc'.format(sat_name) 
    # file_out=dir_out+'DWD_{}.nc'.format(sat_name) 
    # file_out=dir_out+'AMM15_{}.nc'.format(sat_name) 
    file_out=dir_out+'EIS2023_{}.nc'.format(sat_name) 
    
    lon_min = -4.0      ; lon_max = 9.5
    lat_min = 48.6      ; lat_max = 60.4
    
    WAM_data_pair=[]
    SAT_data_pair=[]
    
    yy_s=2022
    yy_e=2024   #2023
    
    # if(sat_name=='S6a'):
    #     yy_s=2021
        
    years=np.arange(yy_s,yy_e) #2023
    
    # yy=2022
    
    for yy in years:
    # for mm in list_mon:
        # print(yy)
       
        # path_yy='{}{}/*.nc'.format(path_file,yy)
        path_yy='{}/*.nc'.format(path_file)
        # path_yy='{}{}/*.nc'.format(path_file,mm)
        
        if all_BS == False:
            ds=xr.open_mfdataset(path_yy) #,combine='by_coords')
            # ds_tmp = ds.VHM0
            # ds_WBS=ds_tmp.resample(time="3H").nearest(tolerance='3m')            #### take every 3H data to the same with GLOBAL data
            # ds_tmp = ds.VHM0
            # ds_WBS = ds_tmp.rename({"latitude":"lat","longitude":"lon"})        #### for GLOBAL DATA rename lat, lon to avoid conflig
            ds_WBS = ds.VHM0
            ds.close()
        else:
            
            ds=xr.open_mfdataset(path_yy) #,combine='by_coords') 
            ds_HS_ALL=ds.VHM0
            ds_HS_ALL.rio.set_crs("epsg:4326") 
            # extract data in WBS
            ds_WBS = ds_HS_ALL.rio.clip_box(
                    minx=lon_min,
                    miny=lat_min,
                    maxx=lon_max,
                    maxy=lat_max,
                    crs="EPSG:4326",
                        )    
            ds.close();ds_HS_ALL.close()
        
        path_sat='{}NOS_Filt_{}_{}.txt'.format(path_alt,sat_name,yy)
        print(path_sat)
        
        with open(path_sat,'r') as ff:
            lines=ff.readlines()
        ff.close()
        
        # num_file=1
        num_file=len(lines)
        
        for kk in range(num_file):
            file_sat=lines[kk][:-1]
            ds_sat=xr.open_dataset(file_sat)
            lon=ds_sat.longitude
            lat=ds_sat.latitude
            tid=ds_sat.time
            
            if (tid[0]>=ds_WBS.time[0] and tid[-1]<=ds_WBS.time[-1]):               #### check if SAT time in range of MODEL DATA
                print('do collocation...')
                # print(tid[0])
                ii_bs=np.where((lat>lat_min) & (lat<lat_max) &(lon>lon_min) & (lon<lon_max))[0]
                nn=len(ii_bs)
    
                if (nn>0):
                    hs_sat=ds_sat.VAVH.isel(time=ii_bs).values
                    lon_sat=ds_sat.longitude.isel(time=ii_bs)
                    lat_sat=ds_sat.latitude.isel(time=ii_bs)
                    tid_sat=ds_sat.time.isel(time=ii_bs)
                
                    ds_HS=ds_WBS.interp(time=tid_sat)
                    
                    hs_wam=ds_HS.sel(lat=lat_sat,lon=lon_sat,method='nearest').values
            
                    SAT_data_pair=np.append(SAT_data_pair,hs_sat)
                    WAM_data_pair=np.append(WAM_data_pair,hs_wam) 
            # else:
            #     print('sat data is outside the range of model data')    
           
        #### close data
        ds_WBS.close()
        
    nx=SAT_data_pair.shape[0]
    
    ##### save to nc file
    rootgrp = netCDF4.Dataset(file_out, "w", CLOBBER='true', format="NETCDF4")
    
    rootgrp.createDimension("nx", nx)
    
    mod_hs=rootgrp.createVariable("mod_hs","float32",("nx",))
    mod_hs.fill_value=-999.0
    mod_hs.missing_value=-999.0
    mod_hs.units='m'
    
    alt_hs=rootgrp.createVariable("alt_hs","float32",("nx",))
    alt_hs.fill_value=-999.0
    alt_hs.missing_value=-999.0
    alt_hs.units='m'
    
    mod_hs[:]=WAM_data_pair[:]
    alt_hs[:]=SAT_data_pair[:]
    
    rootgrp.close()
   
###################################################################            
t2 = datetime.now()
end_time = t2.strftime("%H:%M:%S")
print("End Time =", st_time)

t=t2-t1
print('Time needed {}'.format(t))
