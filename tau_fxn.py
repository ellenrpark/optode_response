#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:57:09 2024

@author: epark
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.interpolate import LinearNDInterpolator
from other_fxn import visc
import gsw
from other_fxn import GetL, CalculateSc

def TauEqn(X, A, b, factor):
    V, T = X
    T_scaling = ((298.15-T*factor)/(273.15+T*factor))
    return A/(V+b)*T_scaling

def GetTauFit(V, T, tau, p0=[1000, 10, 5]):
    # Initial guess
    # p0 = [1000, 10, 5]
    coeff, covar = curve_fit(TauEqn,(V, T), tau ,p0)
    
    # Compute standard deviation of parameters
    sigma = np.sqrt(np.diagonal(covar))
    
    return coeff, covar, sigma

def ResponseTimeCorrection(times, filtered_data, tau):
    
    # If method == 'Inverse'
    #   data = unnfilterd  (ex: CTD cast)
    #   corrected_data = filtered (ex: optode)
    # If method == 'Filter'
    #   data = filtered (ex: optode)
    #   corrected_data = unfiltered (ex: response time corrected optode)
    
    # mean_corrected = np.zeros(filtered_data.shape[0]-1)*np.NaN
    
    # for ti in np.arange(mean_corrected.shape[0]):
    #     ti_plus = ti+1
        
    #     dt = times[ti_plus] - times[ti]
        
    #     b = 1/(1+2*tau[ti]/dt)
    #     a = 1-2*b
        
    #     mean_corrected[ti] = (1/(2*b))*(filtered_data[ti_plus]-a*filtered_data[ti])
    
    good_inds = np.where(np.isnan(filtered_data) == False)[0]
    bad_inds =  np.where(np.isnan(filtered_data) == True)[0]
    
    mean_corrected = np.zeros(good_inds.shape[0]-1)*np.NaN
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
    
    #mean_times = (times[1:]+times[:-1])/2

    corrected_data=np.interp(times, mean_times, mean_corrected)
    
    # Add Nan's back
    corrected_data[bad_inds]=np.NaN
    
    return corrected_data

def InverseResponseTimeCorrection(all_times, all_insitu_data, all_filt_data, time_window=3,
                                  solve_fr = False):
    
    # remove nan's
    good_inds = np.where((np.isnan(all_insitu_data) == False) & (np.isnan(all_filt_data) == False))[0]
    bad_inds =  np.where((np.isnan(all_insitu_data) == True) | (np.isnan(all_filt_data) == False))[0]
    
    times = all_times[good_inds]
    insitu_data = all_insitu_data[good_inds]
    filt_data = all_filt_data[good_inds]
    
    max_window = int(np.ceil(times.shape[0]/time_window))*time_window+1
    windows = np.arange(1, max_window+time_window, time_window)
    
    # Set last value as final index
    windows[-1] = times.shape[0]
    
    # 5 times the mean time window
    
    tau_list = np.arange(5, 200)
    f0_list = np.arange(-5, 5.1,0.1)
    r_list = np.arange(-5, 5.1,0.1)

    
    # Initial guesses
    f0 = 0; r = 0;
    
    # Get optimal tau
    taus = np.zeros(insitu_data.shape[0])*np.NaN
    corrected_data = np.zeros(insitu_data.shape[0])*np.NaN
    corrected_data[0] = (insitu_data[0]+r)-f0
    print('Getting initial taus')
    for wi in np.arange(windows.shape[0]-1):
        
        inds = np.arange(windows[wi], windows[wi+1])
        
        best_tau = np.NaN
        best_rmse  = np.inf
        
        for ti in np.arange(len(tau_list)):
            tau = tau_list[ti]
        
            for xi in np.arange(inds.shape[0]):
    
                dt = times[inds[xi]]- times[inds[xi]-1]
                
                if dt != 0:
                    
                    b = 1/(1+2*tau/dt)
                    a = 1-2*b
                
                    corrected_data[inds[xi]] = a*corrected_data[inds[xi]-1] + \
                        b*((insitu_data[inds[xi]]+r) + (insitu_data[inds[xi]-1]+r))
                    
                else:
                    # Change in time is zero...use previous time
                    corrected_data[inds[xi]] = corrected_data[inds[xi]-1]
                    
            # Calculate rmse for given tau
            rmse = np.sqrt(np.nanmean((corrected_data[inds]-filt_data[inds])**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_tau = tau
        
        # Save best_rmse 
        taus[inds] = best_tau
    
    if solve_fr == True:
        # Set first tau
        taus[0] = taus[1]
        
        # Smooth taus
        N = time_window*3
        y_padded = np.pad(taus, (N//2, N-1-N//2), mode='edge')
        taus = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
    
        # Use optimal taus to  refine f0 and r
        print('Finding optimal f0 and r')
        rmse = np.zeros((f0_list.shape[0], r_list.shape[0]))*np.NaN
            
        for fi in np.arange(f0_list.shape[0]):
            f0 = f0_list[fi]
            
            for ri in np.arange(r_list.shape[0]):
                r = r_list[ri]
                
                corrected_data = np.zeros(insitu_data.shape[0])*np.NaN
                corrected_data[0] = (insitu_data[0]+r)-f0
                
                for wi in np.arange(windows.shape[0]-1):
                    
                    inds = np.arange(windows[wi], windows[wi+1])
                    tau = taus[inds]
                    
                    for xi in np.arange(inds.shape[0]):
            
                        dt = times[inds[xi]]- times[inds[xi]-1]
                        
                        if dt != 0:
                            
                            b = 1/(1+2*tau[xi]/dt)
                            a = 1-2*b
                        
                            corrected_data[inds[xi]] = a*corrected_data[inds[xi]-1] + \
                                b*((insitu_data[inds[xi]]+r) + (insitu_data[inds[xi]-1]+r))
                            
                        else:
                            # Change in time is zero...use previous time
                            corrected_data[inds[xi]] = corrected_data[inds[xi]-1]
                            
                # Calculate rmse for given tau
                rmse[fi, ri] = np.sqrt(np.nanmean((corrected_data-filt_data)**2))
                
        # Optimal
        f0 = f0_list[np.unravel_index(np.nanargmin(rmse), rmse.shape)[0]]
        r = r_list[np.unravel_index(np.nanargmin(rmse), rmse.shape)[1]]
        
        # Re-calculate tau 
        # Get optimal tau
        print('Getting optimal tau')
        taus = np.zeros(insitu_data.shape[0])*np.NaN
        corrected_data = np.zeros(insitu_data.shape[0])*np.NaN
        corrected_data[0] = (insitu_data[0]+r)-f0
        
        for wi in np.arange(windows.shape[0]-1):
            
            inds = np.arange(windows[wi], windows[wi+1])
            
            best_tau = np.NaN
            best_rmse  = np.inf
            
            for ti in np.arange(len(tau_list)):
                tau = tau_list[ti]
            
                for xi in np.arange(inds.shape[0]):
        
                    dt = times[inds[xi]]- times[inds[xi]-1]
                    
                    if dt != 0:
                        
                        b = 1/(1+2*tau/dt)
                        a = 1-2*b
                    
                        corrected_data[inds[xi]] = a*corrected_data[inds[xi]-1] + \
                            b*((insitu_data[inds[xi]]+r) + (insitu_data[inds[xi]-1]+r))
                        
                    else:
                        # Change in time is zero...use previous time
                        corrected_data[inds[xi]] = corrected_data[inds[xi]-1]
                        
                # Calculate rmse for given tau
                rmse = np.sqrt(np.nanmean((corrected_data[inds]-filt_data[inds])**2))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_tau = tau
            
            # Save best_rmse 
            taus[inds] = best_tau
        
    # Set first tau
    taus[0] = taus[1]
    
    # Smooth taus
    N = 10
    y_padded = np.pad(taus, (N//2, N-1-N//2), mode='edge')
    taus = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
        
    # Final correction
    corrected_data = np.zeros(insitu_data.shape[0])*np.NaN
    corrected_data[0] = (insitu_data[0]+r)-f0
    
    for wi in np.arange(windows.shape[0]-1):
        
        inds = np.arange(windows[wi], windows[wi+1])
        tau = taus[inds]
        
        for xi in np.arange(inds.shape[0]):

            dt = times[inds[xi]]- times[inds[xi]-1]
            
            if dt != 0:
                
                b = 1/(1+2*tau[xi]/dt)
                a = 1-2*b
            
                corrected_data[inds[xi]] = a*corrected_data[inds[xi]-1] + \
                    b*((insitu_data[inds[xi]]+r) + (insitu_data[inds[xi]-1]+r))
                
            else:
                # Change in time is zero...use previous time
                corrected_data[inds[xi]] = corrected_data[inds[xi]-1]
    
    final_data = np.zeros(all_times.shape[0])*np.NaN
    final_data[good_inds] = corrected_data
    
    final_tau = np.zeros(all_times.shape[0])*np.NaN
    final_tau[good_inds] = taus
    
    return final_data, final_tau, f0, r

def InverseResponseTimeCorrection_WithParams(all_times, all_insitu_data, all_filt_data,
                                             all_taus, f0, r):
    
    # remove nan's
    good_inds = np.where((np.isnan(all_insitu_data) == False) & (np.isnan(all_filt_data) == False))[0]
    bad_inds =  np.where((np.isnan(all_insitu_data) == True) | (np.isnan(all_filt_data) == False))[0]
    
    times = all_times[good_inds]
    insitu_data = all_insitu_data[good_inds]
    filt_data = all_filt_data[good_inds]
    taus =  all_taus[good_inds]
    
    # time_window = 5
    # max_window = int(np.ceil(times.shape[0]/time_window))*time_window+1
    # windows = np.arange(1, max_window, time_window)
    

    corrected_data = np.zeros(insitu_data.shape[0])*np.NaN
    corrected_data[0] = (insitu_data[0]+r)-f0
    
    
    for xi in np.arange(corrected_data.shape[0]-1)+1:
        
        dt = times[xi]- times[xi-1]
        
        if dt != 0:
            
            b = 1/(1+2*taus[xi]/dt)
            a = 1-2*b
        
            corrected_data[xi] = a*corrected_data[xi-1] + \
                b*((insitu_data[xi]+r) + (insitu_data[xi-1]+r))
            
        else:
            # Change in time is zero...use previous time
            corrected_data[xi] = corrected_data[xi-1]
                    
    final_data = np.zeros(all_times.shape[0])*np.NaN
    final_data[good_inds] = corrected_data

    
    return final_data

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

            
def ReEqn(Re, A, b):
    return A/(Re+b)

def GetReFit(Re, tau, p0=[4427908, 45219]):

    coeff, covar = curve_fit(ReEqn,Re, tau ,p0)
    
    # Compute standard deviation of parameters
    sigma = np.sqrt(np.diagonal(covar))
    
    return coeff, covar, sigma

def CalculateReynolds(L,temp, psal, velocity):
    
    nu = visc(psal, temp)
    Re = (velocity/100)*L/nu
    
    return Re

def FrictionFactor(Re, e_D=0):
    one_over_f = -3.6*np.log10(6.9/Re+(e_D/3.7)**(10/9))
    ff = (1/one_over_f)**2
    
    return ff

def GetResponseTime(sensor, T,S, V, method, fit_type = 'doxy'):
    
    if 'park' in method:
        
        # Temperature must be potential temperature
        
        results = pd.read_csv('results/params/tau_predict_params.csv', index_col = 0)
        
        # Get relevant length scale
        # if method == 'park_flume':
        #     L = GetL('flume')
        # elif method == 'park_float':
        #     L = GetL('float')
        
        # Calculate Reynolds number
        L=1
        Re = CalculateReynolds(L, T, S, V)
        
        # Calculate Sc number
        # Sc = CalculateSc(S, T)
        
        X = Re #*Sc**(2/3)
        
        tau = ReEqn((X), 
                     results.loc[sensor,'A_'+fit_type], 
                     results.loc[sensor,'b_'+fit_type])
                     # results.loc[sensor,'F_'+fit_type])
        
    elif method == 'bittig':
        
        # get bittig tau
        if sensor == 'AA4330':
            bittig = pd.read_csv('csv/bittig_update_sup/T_lL_tau_3830_4330.csv', header = 0,
                                 index_col = 0)
            
        elif sensor == 'SBE63':
            bittig = pd.read_csv('csv/bittig_update_sup/T_lL_tau_SBE63.csv', header = 0,
                                 index_col = 0)
            
        
        bittig.columns = bittig.columns.values.astype(int)
        II, TT = np.meshgrid(bittig.columns.values,
                             bittig.index.values)
        
        # 2D interpolation of grid
        interp = LinearNDInterpolator(list(zip(II.flatten(), 
                                               TT.flatten())), 
                                      bittig.values.flatten())
            
        tau = np.zeros(T.shape[0])*np.NaN

        
        IL = ConvertFlowSpeed_lL_Float(V, sensor)
        tau = interp(IL, T)
        
    return tau
    
    
    
    
def TimeLag(times, insitu_data, taus):
    
    lagged_data = np.zeros(insitu_data.shape[0])
    f0 = 0
    
    lagged_data[0]=insitu_data[0]-f0
    
    
    for i in np.arange(insitu_data.shape[0]-1):
        
        dt = times[i+1]- times[i]
        b = 1/(1+2*taus[i]/dt)
        a = 1-2*b
    
        lagged_data[i+1] = a*lagged_data[i] +  b*(insitu_data[i+1] + insitu_data[i])
        
    return lagged_data
