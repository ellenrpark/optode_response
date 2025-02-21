#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:57:09 2024

@author: epark
"""

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from gasex.phys import visc


def ResponseTimeCorrection(times, filtered_data, tau):
    
    good_inds = np.where(np.isnan(filtered_data) == False)[0]
    bad_inds =  np.where(np.isnan(filtered_data) == True)[0]
    
    mean_corrected = np.zeros(good_inds.shape[0]-1)*np.nan
    mean_times = mean_corrected.copy()
    
    for ii in np.arange(mean_corrected.shape[0]):
        ti = good_inds[ii]
        ti_plus = good_inds[ii+1]
        
        dt = times[ti_plus] - times[ti]
        
        if dt != 0:
            
            b = 1/(1+2*tau[ti]/dt)
            a = 1-2*b
            
            mean_corrected[ii] = (1/(2*b))*(filtered_data[ti_plus]-a*filtered_data[ti])
            mean_times[ii] = (times[ti]+times[ti_plus])/2
    
    corrected_data=np.interp(times, mean_times, mean_corrected)
    
    # Add Nan's back
    corrected_data[bad_inds]=np.nan
    
    return corrected_data


def ConvertFlowSpeed_lL_Float(V, sensor):
    
    if sensor == 'AA4330':
        
        # Equation 1 Bittig and Kortzinger, 2017
        if V.shape == ():
            if V <= 9.5:
                il = 210 - 110/0.095*np.abs(V/100)
            else:
                il = 20 + 80/0.905*(1-np.abs(V/100))
        else:
            
            il = np.where(V<=9.5, 
                          210 - 110/0.095*np.abs(V/100),
                          20 + 80/0.905*(1-np.abs(V/100)))
    elif sensor == 'SBE63':
        
        # Equation A4 Bittig and Kortzinger, 2017
        # Pumped flow rate = 600 mL/min
        V_dot = 600
        il = 1.8e4/(V_dot)+4*np.ones(V.shape)
        
    return il

            
def TauEqn(Re, A, b):
    return A/(Re+b)


def CalculateReynolds(temp, psal, velocity, L = 1):
    
    nu = visc(psal, temp)
    Re = (velocity/100)*L/nu
    
    return Re


def GetResponseTime(sensor, T,S, V, method, fit_type = 'doxy'):
    
    if 'park' in method:
        
        # Temperature must be potential temperature
        results = pd.read_csv('data/tau_predict_params.csv', index_col = 0)

        # Calculate Reynolds number
        Re = CalculateReynolds(T, S, V)
        
        tau = TauEqn(Re, 
                     results.loc[sensor,'A_'+fit_type], 
                     results.loc[sensor,'b_'+fit_type])
        
    elif method == 'bittig':
        
        # get bittig tau
        if sensor == 'AA4330':
            bittig = pd.read_csv('data/T_lL_tau_3830_4330.csv', header = 0,
                                 index_col = 0)
            
        elif sensor == 'SBE63':
            bittig = pd.read_csv('data/T_lL_tau_SBE63.csv', header = 0,
                                 index_col = 0)
            
        
        bittig.columns = bittig.columns.values.astype(int)
        II, TT = np.meshgrid(bittig.columns.values,
                             bittig.index.values)
        
        # 2D interpolation of grid
        interp = LinearNDInterpolator(list(zip(II.flatten(), 
                                               TT.flatten())), 
                                      bittig.values.flatten())
            
        tau = np.zeros(T.shape[0])*np.nan

        
        IL = ConvertFlowSpeed_lL_Float(V, sensor)
        tau = interp(IL, T)
        
    return tau
    
