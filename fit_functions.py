import math
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import percentile as perc
from sklearn.metrics import mean_squared_error

def find_opt_bins(data_drop, popt, x_lim, bin_min, bin_max):
    bin_range = np.arange(bin_min,bin_max,1)
    mse=[]
    for i in bin_range:
        #### Define x space and number of bins
        bins_hist = np.linspace(0, x_lim, i)
        #### Calculate weights for histogram normalization
        weights_drop = np.ones_like(data_drop)/len(data_drop)
        
        data_entries, bins = np.histogram(data_drop, weights=weights_drop, bins=bins_hist, normed=True)
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        fit_entries = sp.stats.erlang.pdf(binscenters, *popt)
        mse.append(mean_squared_error(data_entries,fit_entries))
    
    return bin_range[mse.index(min(mse))]

#%%
def drop_erlang_outliers(data, popt):
    data_drop = data.copy()
    n_dropped = 1
    while n_dropped > 0:
        if popt[0]<1:
            istrue = data_drop > perc(data_drop,99)
        else:
            istrue1 = data_drop > sp.stats.erlang.mean(*popt)+3.4*sp.stats.erlang.std(*popt)
            istrue2 = data_drop < sp.stats.erlang.mean(*popt)-2.4*sp.stats.erlang.std(*popt)
            istrue  = istrue1 + istrue2
            
        data_drop.drop(data_drop.loc[istrue].index,inplace=True)
        popt = sp.stats.erlang.fit(data_drop)
        n_dropped = len(istrue[istrue==True])
    
    return data_drop, popt

#%%
def hist_erlang_fit(data, color, label, spec, unit, ax):
    #### MODIFICATIONS ####
    popt            = sp.stats.erlang.fit(data)
    data_drop, popt = drop_erlang_outliers(data,popt)
    x_lim           = data_drop.max()*1.4
    bin_min         = int(round(10 * x_lim/(5.8*sp.stats.erlang.std(*popt))))
    bin_max         = int(round(20 * x_lim/(5.8*sp.stats.erlang.std(*popt))))
    
    n_bins = find_opt_bins(data_drop, popt, x_lim, bin_min, bin_max)
    bins_hist = np.linspace(0, x_lim, n_bins)
    #### Calculate weights for histogram normalization
    weights = np.ones_like(data)/len(data_drop)
    weights_drop = np.ones_like(data_drop)/len(data_drop)
    #### Plot histograms
    ax.hist(data,      bins=bins_hist, weights=weights,      color=color, alpha=0.4, normed=True)
    ax.hist(data_drop, bins=bins_hist, weights=weights_drop, color=color, alpha=0.4, normed=True, label=label + spec)
    
    xspace = np.linspace(0, x_lim, 1000)    
    ax.plot(xspace, sp.stats.erlang.pdf(xspace, *popt), '-', color='black', lw=3)
    ax.axvline(x=sp.stats.erlang.mean(*popt),                                ymin=0, ymax=1, ls='--', color='black', label='mean=%1.1f'%(sp.stats.erlang.mean(*popt)) + unit, lw=2)
    ax.axvline(x=sp.stats.erlang.mean(*popt)+3.4*sp.stats.erlang.std(*popt),   ymin=0, ymax=1, ls='--', color='black', label='3.4x std=%1.1f'%(sp.stats.erlang.std(*popt)*3.4) + unit)
    ax.axvline(x=sp.stats.erlang.mean(*popt)-2.4*sp.stats.erlang.std(*popt), ymin=0, ymax=1, ls='--', color='black', label='2.4x std=%1.1f'%(sp.stats.erlang.std(*popt)*2.4) + unit)
    ax.legend(framealpha=1)
    
    return data_drop, popt, x_lim

#%%
def gaussian(x,a,b,c):
    return a*np.exp(-((x-b)**2/(2*c**2)))

#%%
def poisson(k,lamb):
    return sp.stats.poisson.pmf(k, lamb)

#%%
def hist_gauss_fit(data, n_bins, color, label, unit, xstd, ax):
    #### MODIFICATIONS ####
    p_initial=(1,data.mean(),data.std())
    x_lim_low = data.min()
    x_lim_up  = data.max()
     
    #### Define x space and number of bins
    bins_hist = np.linspace(x_lim_low, x_lim_up, n_bins)
    #### Calculate weights for histogram normalization
    weights            = np.ones_like(data)/len(data)
    data_entries, bins = np.histogram(data, weights=weights, bins=bins_hist)
    #### Fit to the histogram
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    popt, pcov = sp.optimize.curve_fit(gaussian, xdata=binscenters, ydata=data_entries, p0=p_initial,maxfev=10000)
    #### Plot histogram, fit and mean
    xspace = np.linspace(popt[1]-5*popt[2], popt[1]+5*popt[2], 1000)    
    ax.hist(data, bins=bins_hist, weights=weights, color=color, alpha=.5, label=label)
    ax.plot(xspace, gaussian(xspace, *popt), '-', color='black', lw=3)
    ax.axvline(x=popt[1], ymin=0, ymax=1, ls='--', color='black', label='mean = %1.1f '%(popt[1]) + unit, lw=3)
    ax.axvline(x=popt[1]+xstd*popt[2], ymin=0, ymax=1, ls='--', color='black', label='%1dx std = %1.1f '%(xstd, popt[2]*xstd) + unit, lw=2)
    ax.axvline(x=popt[1]-xstd*popt[2], ymin=0, ymax=1, ls='--', color='black', lw=2)
    ax.legend()
    plt.xlim(x_lim_low,x_lim_up)
    
    return popt, xstd

#%%
def hist_poisson_fit(data, n_bins, color, label, unit, xstd, ax):
    #### MODIFICATIONS ####
    x_lim_low = 0
    x_lim_up  = data.max()
     
    #### Define x space and number of bins
    # bins_hist = np.linspace(x_lim_low, x_lim_up, n_bins)
    bins_hist = np.arange(0,x_lim_up,10)
    #### Calculate weights for histogram normalization
    weights            = np.ones_like(data)/len(data)
    data_entries, bins = np.histogram(data, bins=bins_hist)
    #### Fit to the histogram
    binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    popt, pcov = sp.optimize.curve_fit(poisson, xdata=binscenters, ydata=data_entries,  p0=100)
    #### Plot histogram, fit and mean
    xspace = np.arange(0, 500) 
    ax.hist(data, bins=bins_hist, weights=weights, color=color, alpha=.5, label=label)
    ax.plot(xspace, poisson(xspace, *popt)*5, '-', color='black', lw=3)
    ax.axvline(x=popt[0], ymin=0, ymax=1, ls='--', color='black', label='mean = %1.1f '%(popt[0]) + unit, lw=3)
    # ax.axvline(x=popt[0], ymin=0, ymax=1, ls='--', color='black', label='%1dx std = %1.1f '%(xstd, popt[1]*xstd) + unit, lw=2)
    # ax.axvline(x=popt[0], ymin=0, ymax=1, ls='--', color='black', lw=2)
    ax.legend()
    plt.xlim(x_lim_low,x_lim_up)
    
    return popt, xstd

