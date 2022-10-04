#################################################### Load packages
import os #platform independent paths
import pandas as pd
import numpy as np
import scipy as sp
import math as math
import matplotlib.pyplot as plt
### Load an extra custom file
import sys
# sys.path.insert(1, 'C:/Julian_Bauer/lbFCS/MP')
sys.path.insert(1, '...directory of files')
import fit_functions as ff
import io

#%%
#################################
'''Filter for DNA PAINT traces'''
#################################
def stat_filter(MP_locs,frames):
    ### copy all localizations to locs_drop which will than be manipulated by filtering
    MP_locs_drop=MP_locs.copy()
    
    ### mean_frames: drop all traces that deviate more than 40% from Frames/2
    istrue_mf=np.abs(MP_locs_drop.frame-frames*0.5)/(frames*0.5)>0.35
    ### std_frames: drop 20% of the traces with the lowest std
    istrue_sf=MP_locs_drop.std_frame<(0.7*MP_locs_drop.std_frame.mean())
    ### n_events: drop all traces with less than x events
    istrue_ne_down=MP_locs_drop.n_events<5
    ### kinetics: drop all traces where mono_tau and mono_tau_lin deviate more than x %
    istrue_kin=(np.abs(MP_locs_drop.mono_tau_lin-MP_locs_drop.mono_tau)/MP_locs_drop.mono_tau)>0.5
    
    istrue = istrue_mf + istrue_sf + istrue_ne_down + istrue_kin
    MP_locs_drop.drop(MP_locs_drop.loc[istrue].index,inplace=True)
    
    return MP_locs_drop

#%%
########################
'''plot filtered DATA'''
########################

def stat_plot(MP_locs,data_set,frames,save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    
    f=plt.figure(num=1,figsize=[8,6])
    f.subplots_adjust(bottom=0.1,top=0.95,left=0.1,right=0.95)
    f.clear()
    
    bins_mean = np.linspace(0,frames,50)
    bins_std  = np.linspace(0,frames/2,50)
    bins_taub = np.linspace(0,4,50)
    bins_n    = np.linspace(0,300,50)
    
    ax=f.add_subplot(2,2,1)
    ax.hist(MP_locs.frame, bins=bins_mean, color='grey', alpha=.3)
    ax.hist(data_set.frame, bins=bins_mean)
    plt.xlim(0,frames)
    ax.set_xlabel('mean frame [#]')
    
    ax=f.add_subplot(2,2,3)
    ax.hist(MP_locs.std_frame, bins=bins_std, color='grey', alpha=.3)
    ax.hist(data_set.std_frame, bins=bins_std)
    plt.xlim(0,frames/2)
    ax.set_xlabel('std frame [#]')
    
    ax=f.add_subplot(2,2,2)
    ax.hist((np.abs(MP_locs.mono_tau_lin-MP_locs.mono_tau)/MP_locs.mono_tau), bins=bins_taub, color='grey', alpha=.3)
    ax.hist((np.abs(data_set.mono_tau_lin-data_set.mono_tau)/data_set.mono_tau), bins=bins_taub)
    plt.xlim(0,2)
    ax.set_xlabel('(mono_tau_lin - mono_tau) / mono_tau')
    
    ax=f.add_subplot(2,2,4)
    ax.hist(MP_locs.n_events, bins=bins_n, color='grey', alpha=.3)
    ax.hist(data_set.n_events, bins=bins_n)
    plt.xlim(0,300)
    ax.set_xlabel('number of events [#]')
    
    plt.savefig(save_path + '_stats')

#%%
######################################
'''assign spots & drop single spots'''
######################################
# data_set_assigned, picks = ff.spot_assignment(data_set)
def spot_assignment(data_set, d_min=1.05, d_max=1.3):
    from tqdm import tqdm
    picks = pd.DataFrame(columns=['x','y','pair'])
    data_set['pair'] = float('NaN')
    data_set['mono_tau_diff'] = float('NaN')
    data_set['distance'] = float('NaN')
    
    p,t,d=len(data_set.columns)-3,len(data_set.columns)-2,len(data_set.columns)-1

    ind=0
    for i in tqdm(range(0, len(data_set.x)-1)):
        k=i+1
        while data_set.iloc[k].y < data_set.iloc[i].y+d_max:
            dist = math.sqrt((data_set.iloc[i].x-data_set.iloc[k].x)**2+(data_set.iloc[i].y-data_set.iloc[k].y)**2)
            if d_min < dist < d_max:
                picks.loc[ind,['x','y']] = [data_set.iloc[i].x-(data_set.iloc[i].x-data_set.iloc[k].x)/2,data_set.iloc[i].y-(data_set.iloc[i].y-data_set.iloc[k].y)/2] 
                if math.isnan(data_set.iloc[i,p]):
                    tau_diff = abs(data_set.iloc[i,3]-data_set.iloc[k,3])
                    data_set.iloc[i,[p,t,d]] = [ind,tau_diff,dist]
                    data_set.iloc[k,[p,t,d]] = [ind,tau_diff,dist]
                    picks.loc[ind,'pair'] = ind
                else:
                    data_set.iloc[k,p] = data_set.iloc[i,p]
                    picks.loc[ind,'pair'] = data_set.iloc[i,p]
                ind+=1
            if k < len(data_set.x)-1:
                k+=1 
            else:
                break
            
    data_set_assigned = data_set.groupby('pair').filter(lambda x : 1<len(x)<3)
    
    return data_set_assigned, picks

#%%
###########################################################
'''save assigned spots (pick per spot & pick per origami'''
###########################################################
def picks_save(data_set_assigned,picks,path):
    import lbfcs.picasso_wrap as pic_wrap
    oversampling=1
    
    #### Save all filtered single picks
    single_picks=data_set_assigned[['x','y']]
    single_picks.index = np.arange(0,len(single_picks.index),1)
    pick_diameter = 1
    io._save_picks(pic_wrap._coordinate_convert(single_picks, (0,0), oversampling),
                         pick_diameter,
                         os.path.splitext(path)[0]+'_filterpicks.yaml')
    
    #### Save all paired picks
    pair_picks = picks.groupby('pair').filter(lambda x : len(x)<2)
    pair_picks.index = np.arange(0,len(pair_picks.index),1)
    pick_diameter = 2
    io._save_picks(pic_wrap._coordinate_convert(pair_picks, (0,0), oversampling),
                         pick_diameter,
                         os.path.splitext(path)[0]+'_pairpicks.yaml')

#%%
###########################################
'''plot spot distance & filter out 3xstd'''
###########################################
def distance_plot(data_set_assigned,label,save_path):
    n_bins = 15
    color='firebrick'
    unit='nm'
    xstd=3
    
    f=plt.figure(num=2,figsize=[5,3])
    f.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.95)
    f.clear()
    
    ax=f.add_subplot(111)
    popt, xstd = ff.hist_gauss_fit(data_set_assigned.distance*100, n_bins, color, label, unit, xstd, ax)
    plt.xlim(95,135)
    ax.set_xlabel('spot distance [nm]')
    plt.savefig(save_path + '_spot_distance')
    
    data_set_assigned_dist_filtered = data_set_assigned.copy()
    istrue1 = data_set_assigned_dist_filtered.distance*100 > popt[1]+3*popt[2]
    istrue2 = data_set_assigned_dist_filtered.distance*100 < popt[1]-3*popt[2]
    istrue = istrue1 + istrue2
    data_set_assigned_dist_filtered.drop(data_set_assigned_dist_filtered.loc[istrue].index,inplace=True)
    ##########################################
    '''save data_set_assigned_dist_filtered'''
    ##########################################
    return data_set_assigned_dist_filtered

#%%
#####################
''' PHOTONS values'''
#####################
def photon_plot(data_set_assigned_dist_filtered,label,save_path):
    n_bins = 10
    color='black'
    unit='ph'
    xstd=3
    
    f=plt.figure(num=2,figsize=[5,3])
    f.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.95)
    f.clear()
    ax=f.add_subplot(111)
    popt, xstd = ff.hist_gauss_fit(data_set_assigned_dist_filtered.photons, n_bins, color, label, unit, xstd, ax)
    plt.xlim(0,data_set_assigned_dist_filtered.photons.max()*1.3)
    ax.set_xlabel('mean photons [#]')
    
    plt.savefig(save_path + '_photons')
#%%
#########################################################################
'''create characteristic values (modification, reference & normalized)'''
#########################################################################
# data_sec_initial = data_set_assigned_dist_filtered.groupby('pair', as_index=False).apply(lambda df: ff.bin_ds(df, sec=True, CycleTime=CycleTime))#, assign_crit='n_events'))
# data_fr_initial  = data_set_assigned_dist_filtered.groupby('pair', as_index=False).apply(lambda df: ff.bin_ds(df, sec=False, CycleTime=CycleTime))

def bin_ds(df,sec,CycleTime,assign_crit='mono_taub'):
    # index = int(df.pair.mean())
    # print(df)
    s_out = pd.DataFrame(data= np.zeros([1,13]), columns=['pair','distance','ref_taub','mod_taub','mod_taub_norm','taub_diff','taub_diff_norm','ref_taud','mod_taud','ref_n_events','mod_n_events','ref_n_locs','mod_n_locs'])
    # print(s_out)
    s_out[['pair','distance']] = df[df['mono_taub'] == df['mono_taub'].min()][['pair','distance']].values
    
    if assign_crit == 'n_events':
        # print(df[assign_crit].min(),df[assign_crit].max())
        try:
            s_out[['ref_taub','ref_taud','ref_n_events','ref_n_locs']] = df[df[assign_crit] == df[assign_crit].max()][['mono_taub','mono_taud','n_events','n_locs']].values
            s_out[['mod_taub','mod_taud','mod_n_events','mod_n_locs']] = df[df[assign_crit] == df[assign_crit].min()][['mono_taub','mono_taud','n_events','n_locs']].values
        except:
            s_out[['ref_taub','ref_taud','ref_n_events','ref_n_locs']] = df[df['mono_taub'] == df['mono_taub'].max()][['mono_taub','mono_taud','n_events','n_locs']].values
            s_out[['mod_taub','mod_taud','mod_n_events','mod_n_locs']] = df[df['mono_taub'] == df['mono_taub'].min()][['mono_taub','mono_taud','n_events','n_locs']].values
            print(1)
    else:
        s_out[['ref_taub','ref_taud','ref_n_events','ref_n_locs']] = df[df[assign_crit] == df[assign_crit].min()][['mono_taub','mono_taud','n_events','n_locs']].values
        s_out[['mod_taub','mod_taud','mod_n_events','mod_n_locs']] = df[df[assign_crit] == df[assign_crit].max()][['mono_taub','mono_taud','n_events','n_locs']].values
    
    if sec:
        s_out[['ref_taub','ref_taud','mod_taub','mod_taud']] = s_out[['ref_taub','ref_taud','mod_taub','mod_taud']]*CycleTime
   
    s_out[['mod_taub_norm']] = s_out.mod_taub / s_out.ref_taub
    s_out['n_locs_diff'] = s_out.mod_n_locs - s_out.ref_n_locs
    s_out['n_events_diff'] = s_out.ref_n_events - s_out.mod_n_events
    s_out['taud_diff'] = s_out.mod_taud - s_out.ref_taud
    s_out['taub_diff'] = s_out.mod_taub - s_out.ref_taub
    s_out['taub_diff_norm'] = (s_out.mod_taub - s_out.ref_taub)/s_out.ref_taub
    s_out['taub_ind_norm'] = s_out.mod_taub / s_out.ref_taub

    return s_out

#%%
##################################################
'''general outlier filtering by 3x gaussian std'''
##################################################
def outlier_plot(data_sec_initial, label, save_path):
    data_names=['taud_diff','taub_diff','n_locs_diff','mod_n_locs','ref_n_locs','n_events_diff','mod_n_events','ref_n_events']
    plt_nr=[1,2,4,5,6,7,8,9]
    units =['s','s','#','#','#','#','#','#']
    xstd=3
    n_bins=30
    
    f=plt.figure(num=3,figsize=[15,10])
    f.subplots_adjust(bottom=0.07,top=0.95,left=0.07,right=0.95,hspace=0.25)
    f.clear()
    istrues=[]
    for i in range(0,len(data_names)):    
        ax=f.add_subplot(3,3,plt_nr[i])
        #### MODIFICATIONS ####
        data = data_sec_initial[data_names[i]]
        popt, xstd = ff.hist_gauss_fit(data, n_bins, 'darkblue', label, units[i], xstd, ax)
        # popt, xstd = ff.hist_poisson_fit(data, n_bins, 'darkblue', label, units[i], xstd, ax)
        ax.set_xlabel(data_names[i] + ' [' + units[i] + ']', size=12)
    
        istrues.append(data > popt[1]+xstd*popt[2])
        istrues.append(data < popt[1]-xstd*popt[2])
        
    plt.savefig(save_path + '_outlier_filter')
    data_sec = data_sec_initial.copy()
    # data_fr  = data_fr_initial.copy()
    ### Combine True values and drop data
    istrue = istrues[0] + istrues[1] + istrues[2] + istrues[3] + istrues[4] + istrues[5] + istrues[6] + istrues[7] + istrues[8] + istrues[9] + istrues[10] + istrues[11] + istrues[12] + istrues[13] + istrues[14] + istrues[15] 
    data_sec.drop(data_sec.loc[istrue].index,inplace=True)
    # data_fr.drop(data_fr.loc[istrue].index,inplace=True)
    
    return data_sec

#%%
def save_dict_to_file(dic, save_path):
    with open(save_path + '_value_dict.csv', "w") as f:
        wr = csv.writer(f,delimiter=":")
        wr.writerows(dic.items())
        
#%%
def kinetic_plots_values(data_sec, label, save_path):
    #################################
    '''dark time plot + erlang fit'''
    #################################
    taud_mod = data_sec.mod_taud
    taud_ref = data_sec.ref_taud
    
    f_all=plt.figure(num=100,figsize=[15,10])
    f_all.subplots_adjust(bottom=0.07,top=0.97,left=0.07,right=0.97)
    f_all.clear()
    
    f=plt.figure(num=4,figsize=[6,4])
    f.subplots_adjust(bottom=0.15,top=0.95,left=0.1,right=0.95,hspace=0)
    f.clear()
    
    ax=f.add_subplot(211)
    ax_all=f_all.add_subplot(331)
    taud_mod_drop, popt_mod, x_lim_mod = ff.hist_erlang_fit(taud_mod, 'darkgreen', label, ' mod', 's', ax)
    taud_mod_drop, popt_mod, x_lim_mod = ff.hist_erlang_fit(taud_mod, 'darkgreen', label, ' mod', 's', ax_all)
    ax.set_xticks([])
    ax.set_xlim(0,x_lim_mod)
    ax_all.set_xlim(0,x_lim_mod)
    
    ax=f.add_subplot(212)
    ax_all=f_all.add_subplot(334)
    taud_ref_drop, popt_ref, x_lim_ref = ff.hist_erlang_fit(taud_ref, 'grey', label, ' ref', 's', ax)
    taud_ref_drop, popt_ref, x_lim_ref = ff.hist_erlang_fit(taud_ref, 'grey', label, ' ref', 's', ax_all)
    ax.set_xlim(0,x_lim_mod)
    ax_all.set_xlim(0,x_lim_ref)
    
    ax.set_xlabel('dark time [s]', size=12)
    ax_all.set_xlabel('dark time [s]', size=12)
    f.savefig(save_path + '_dark_time')
    
    value_dict = {'tauD-sample_size'        : len(taud_mod),
                  'tauD-sample_size_erlang' : len(taud_mod_drop),
                  
                  'tauD_mod-mean'           : taud_mod.mean(),
                  'tauD_mod-median'         : taud_mod.median(),
                  'tauD_mod-std'            : taud_mod.std(),
                  'tauD_mod-mean_erlang'    : sp.stats.erlang.mean(*popt_mod),
                  'tauD_mod-median_erlang'  : sp.stats.erlang.median(*popt_mod),
                  'tauD_mod-std_erlang'     : sp.stats.erlang.std(*popt_mod),
                 
                  'tauD_ref-sample_size'    : len(taud_ref_drop),
                  'tauD_ref-mean'           : taud_ref.mean(),
                  'tauD_ref-median'         : taud_ref.median(),
                  'tauD_ref-std'            : taud_ref.std(),
                  'tauD_ref-mean_erlang'    : sp.stats.erlang.mean(*popt_ref),
                  'tauD_ref-median_erlang'  : sp.stats.erlang.median(*popt_ref),
                  'tauD_ref-std_erlang'     : sp.stats.erlang.std(*popt_ref),
                 
                  'tauD_norm-mean'          : taud_mod.mean()     / taud_ref.mean(),
                  'tauD_norm-median'        : taud_mod.median()   / taud_ref.median(),
                  'tauD_norm-std'           : (taud_mod.median()/taud_ref.median())*(((taud_mod.std()/(taud_mod.median()))**2+(taud_ref.std()/(taud_ref.median()))**2)**(1/2)),
                  'tauD_norm-ste'           : (taud_mod.median()/taud_ref.median())*(((taud_mod.std()/(taud_mod.median()))**2+(taud_ref.std()/(taud_ref.median()))**2)**(1/2)) /(len(taud_mod)**(1/2)),
                  'tauD_norm-mean_erlang'   : sp.stats.erlang.mean(*popt_mod)   / sp.stats.erlang.mean(*popt_ref),
                  'tauD_norm-median_erlang' : sp.stats.erlang.median(*popt_mod) / sp.stats.erlang.median(*popt_ref),
                  'tauD_norm-std_erlang'    : (sp.stats.erlang.median(*popt_mod)/sp.stats.erlang.median(*popt_ref))*(((sp.stats.erlang.std(*popt_mod)/(sp.stats.erlang.median(*popt_mod)))**2+(sp.stats.erlang.std(*popt_ref)/(sp.stats.erlang.median(*popt_ref)))**2)**(1/2)), 
                  'tauD_norm-ste_erlang'    : (sp.stats.erlang.median(*popt_mod)/sp.stats.erlang.median(*popt_ref))*(((sp.stats.erlang.std(*popt_mod)/(sp.stats.erlang.median(*popt_mod)))**2+(sp.stats.erlang.std(*popt_ref)/(sp.stats.erlang.median(*popt_ref)))**2)**(1/2)) /(len(taud_mod_drop)**(1/2))    
                 }
    
    ###################################
    '''bright time plot + erlang fit'''
    ###################################
    taub_mod = data_sec.mod_taub
    taub_ref = data_sec.ref_taub
    
    f=plt.figure(num=4,figsize=[6,4])
    f.subplots_adjust(bottom=0.15,top=0.95,left=0.1,right=0.95,hspace=0)
    f.clear()
    
    ax=f.add_subplot(211)
    ax_all=f_all.add_subplot(332)
    taub_mod_drop, popt_mod, x_lim_mod = ff.hist_erlang_fit(taub_mod, 'darkgreen', label, ' mod', 's', ax)
    taub_mod_drop, popt_mod, x_lim_mod = ff.hist_erlang_fit(taub_mod, 'darkgreen', label, ' mod', 's', ax_all)
    ax.set_xticks([])
    ax.set_xlim(0,x_lim_mod)
    ax_all.set_xlim(0,x_lim_mod)
    
    ax=f.add_subplot(212)
    ax_all=f_all.add_subplot(335)
    taub_ref_drop, popt_ref, x_lim_ref = ff.hist_erlang_fit(taub_ref, 'grey', label, ' ref', 's', ax)
    taub_ref_drop, popt_ref, x_lim_ref = ff.hist_erlang_fit(taub_ref, 'grey', label, ' ref', 's', ax_all)
    ax.set_xlim(0,x_lim_mod)
    ax_all.set_xlim(0,x_lim_ref)
    
    ax.set_xlabel('bright time [s]', size=12)
    ax_all.set_xlabel('bright time [s]', size=12)
    f.savefig(save_path + '_bright_time')
        
    tauB_dict = {'tauB-sample_size'       : len(taub_mod),
                 'tauB-sample_size_erlang': len(taub_mod_drop),
                 
                 'tauB_mod-mean'           : taub_mod.mean(),
                 'tauB_mod-median'         : taub_mod.median(),
                 'tauB_mod-std'            : taub_mod.std(),
                 'tauB_mod-mean_erlang'    : sp.stats.erlang.mean(*popt_mod),
                 'tauB_mod-median_erlang'  : sp.stats.erlang.median(*popt_mod),
                 'tauB_mod-std_erlang'     : sp.stats.erlang.std(*popt_mod),
                 
                 'tauB_ref-sample_size'    : len(taub_ref_drop),
                 'tauB_ref-mean'           : taub_ref.mean(),
                 'tauB_ref-median'         : taub_ref.median(),
                 'tauB_ref-std'            : taub_ref.std(),
                 'tauB_ref-mean_erlang'    : sp.stats.erlang.mean(*popt_ref),
                 'tauB_ref-median_erlang'  : sp.stats.erlang.median(*popt_ref),
                 'tauB_ref-std_erlang'     : sp.stats.erlang.std(*popt_ref),
                 
                 'tauB_norm-mean'          : taub_mod.mean()     / taub_ref.mean(),
                 'tauB_norm-median'        : taub_mod.median()   / taub_ref.median(),
                 'tauB_norm-std'           : (taub_mod.median()/taub_ref.median())*(((taub_mod.std()/(taub_mod.median()))**2+(taub_ref.std()/(taub_ref.median()))**2)**(1/2)),                 
                 'tauB_norm-ste'           : (taub_mod.median()/taub_ref.median())*(((taub_mod.std()/(taub_mod.median()))**2+(taub_ref.std()/(taub_ref.median()))**2)**(1/2)) /(len(taub_mod)**(1/2)),                 
                 'tauB_norm-mean_erlang'   : sp.stats.erlang.mean(*popt_mod)   / sp.stats.erlang.mean(*popt_ref),
                 'tauB_norm-median_erlang' : sp.stats.erlang.median(*popt_mod) / sp.stats.erlang.median(*popt_ref),
                 'tauB_norm-std_erlang'    : (sp.stats.erlang.median(*popt_mod)/sp.stats.erlang.median(*popt_ref))*(((sp.stats.erlang.std(*popt_mod)/(sp.stats.erlang.median(*popt_mod)))**2+(sp.stats.erlang.std(*popt_ref)/(sp.stats.erlang.median(*popt_ref)))**2)**(1/2)),
                 'tauB_norm-ste_erlang'    : (sp.stats.erlang.median(*popt_mod)/sp.stats.erlang.median(*popt_ref))*(((sp.stats.erlang.std(*popt_mod)/(sp.stats.erlang.median(*popt_mod)))**2+(sp.stats.erlang.std(*popt_ref)/(sp.stats.erlang.median(*popt_ref)))**2)**(1/2)) /(len(taub_mod_drop)**(1/2))    
                 }
    
    value_dict.update(tauB_dict)
    
    ########################################################
    '''bright time difference (mod-ref) plot + erlang fit'''
    ########################################################
    diff_norm = data_sec.taub_diff_norm
    diff      = data_sec.taub_diff
    
    f=plt.figure(num=4,figsize=[6,4])
    f.subplots_adjust(bottom=0.15,top=0.95,left=0.1,right=0.95,hspace=0)
    f.clear()
    
    ax=f.add_subplot(211)
    ax_all=f_all.add_subplot(333)
    diff_norm_drop, popt_ndif, x_lim_diff_norm = ff.hist_erlang_fit(diff_norm, 'darkblue', label, ' (mod-ref)/ref', ' ', ax)
    diff_norm_drop, popt_ndif, x_lim_diff_norm = ff.hist_erlang_fit(diff_norm, 'darkblue', label, ' (mod-ref)/ref', ' ', ax_all)
    ax.set_xticks([])
    ax.set_xlim(0,x_lim_diff_norm)
    ax_all.set_xlim(0,x_lim_diff_norm)
    
    ax=f.add_subplot(212)
    ax_all=f_all.add_subplot(336)
    diff_drop, popt_dif, x_lim_diff = ff.hist_erlang_fit(diff, 'darkgreen', label, ' (mod-ref)', 's', ax)
    diff_drop, popt_dif, x_lim_diff = ff.hist_erlang_fit(diff, 'darkgreen', label, ' (mod-ref)', 's', ax_all)
    ax.set_xlim(0,x_lim_diff_norm)
    ax_all.set_xlim(0,x_lim_diff)
    
    ax.set_xlabel('bright time difference [s] / [x-fold]', size=12)
    ax_all.set_xlabel('bright time difference [s] / [x-fold]', size=12)
    f.savefig(save_path + '_bright_time_difference')
    
    tauB_diff_dict = {'tauB_diff-sample_size'            : len(diff),
                      'tauB_diff-sample_size_erlang'     : len(diff_drop),
                      'tauB_diff_norm-sample_size_erlang': len(diff_norm_drop),

                      'tauB_diff-mean'                   : diff.mean(),
                      'tauB_diff-median'                 : diff.median(),
                      'tauB_diff-std'                    : diff.std(),
                      'tauB_diff-mean_erlang'            : sp.stats.erlang.mean(*popt_dif),
                      'tauB_diff-median_erlang'          : sp.stats.erlang.median(*popt_dif),
                      'tauB_diff-std_erlang'             : sp.stats.erlang.std(*popt_dif),
                      
                      'tauB_diff_norm-mean'              : diff_norm.mean(),
                      'tauB_diff_norm-median'            : diff_norm.median(),
                      'tauB_diff_norm-std'               : diff_norm.std(),
                      'tauB_diff_norm-mean_erlang'       : sp.stats.erlang.mean(*popt_ndif),
                      'tauB_diff_norm-median_erlang'     : sp.stats.erlang.median(*popt_ndif),
                      'tauB_diff_norm-std_erlang'        : sp.stats.erlang.std(*popt_ndif),
                      }
    
    value_dict.update(tauB_diff_dict)
    
    ###########################################################
    '''origami-wise normalized bright time plot + erlang fit'''
    ###########################################################
    ind_norm  = data_sec.taub_ind_norm
    
    f=plt.figure(num=2,figsize=[5,3])
    f.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.95,hspace=0)
    f.clear()
    
    ax=f.add_subplot(111)
    ax_all=f_all.add_subplot(337)
    ind_norm_drop, popt_ind, x_lim_ind_norm = ff.hist_erlang_fit(ind_norm, 'darkblue', label, ' (mod/ref)', '', ax)
    ind_norm_drop, popt_ind, x_lim_ind_norm = ff.hist_erlang_fit(ind_norm, 'darkblue', label, ' (mod/ref)', '', ax_all)
    ax.set_xlim(0,x_lim_ind_norm)
    ax.set_xlabel('individually mormalized bright times [x-fold]', size=12)
    ax_all.set_xlim(0,x_lim_ind_norm)
    ax_all.set_xlabel('individually mormalized bright times [x-fold]', size=12)

    f.savefig(save_path + '_individually_norm_bright_times')
    
    tauB_ind_norm_dict = {'tauB_ind_norm-sample_size'       : len(ind_norm),
                          'tauB_ind_norm-sample_size_erlang': len(ind_norm_drop),
                          
                          'tauB_ind_norm-mean'              : ind_norm.mean(),
                          'tauB_ind_norm-median'            : ind_norm.median(),
                          'tauB_ind_norm-std'               : ind_norm.std(),
                          'tauB_ind_norm-mean_erlang'       : sp.stats.erlang.mean(*popt_ind),
                          'tauB_ind_norm-median_erlang'     : sp.stats.erlang.median(*popt_ind),
                          'tauB_ind_norm-std_erlang'        : sp.stats.erlang.std(*popt_ind)
                          }
    
    value_dict.update(tauB_ind_norm_dict)
    
    #######################################################################
    '''origami-wise normalized number of localizations plot + erlang fit'''
    #######################################################################
    locs_norm        = data_sec.mod_n_locs / data_sec.ref_n_locs
    
    f=plt.figure(num=2,figsize=[5,3])
    f.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.95,hspace=0)
    f.clear()
    
    ax=f.add_subplot(111)
    ax_all=f_all.add_subplot(338)
    locs_norm_drop, popt_locs, x_lim_locs_norm = ff.hist_erlang_fit(locs_norm, 'darkblue', label, ' (mod/ref)', '', ax)
    locs_norm_drop, popt_locs, x_lim_locs_norm = ff.hist_erlang_fit(locs_norm, 'darkblue', label, ' (mod/ref)', '', ax_all)
    ax.set_xlim(0,x_lim_locs_norm)
    ax.set_xlabel('individually mormalized localizations [x-fold]', size=12)
    ax_all.set_xlim(0,x_lim_locs_norm)
    ax_all.set_xlabel('individually mormalized localizations [x-fold]', size=12)
    
    f.savefig(save_path + '_normalized_localizations')
    
    norm_locs_dict = {'n_locs-sample_size'        : len(locs_norm),           
                      'n_locs-sample_size_erlang' : len(locs_norm_drop),
    
                      'n_locs_mod-mean'           : data_sec.mod_n_locs.mean(),
                      'n_locs_mod-median'         : data_sec.mod_n_locs.median(),
                      'n_locs_mod-std'            : data_sec.mod_n_locs.std(),
                      'n_locs_ref-mean'           : data_sec.ref_n_locs.mean(),
                      'n_locs_ref-mediant'        : data_sec.ref_n_locs.median(),
                      'n_locs_ref-std'            : data_sec.ref_n_locs.std(),
                      
                      'n_locs_norm-mean'          : locs_norm.mean(),
                      'n_locs_norm-median'        : locs_norm.median(),
                      'n_locs_norm-std'           : locs_norm.std(),
                      'n_locs_norm-mean_erlang'   : sp.stats.erlang.mean(*popt_locs),
                      'n_locs_norm-median_erlang' : sp.stats.erlang.median(*popt_locs),
                      'n_locs_norm-std_erlang'    : sp.stats.erlang.std(*popt_locs)
                      }
    
    value_dict.update(norm_locs_dict)
    
    ################################################################
    '''origami-wise normalized number of events plot + erlang fit'''
    ################################################################
    events_norm        = data_sec.mod_n_events / data_sec.ref_n_events
    
    f=plt.figure(num=2,figsize=[5,3])
    f.subplots_adjust(bottom=0.2,top=0.95,left=0.1,right=0.95,hspace=0)
    f.clear()
    
    ax=f.add_subplot(111)
    ax_all=f_all.add_subplot(339)
    events_norm_drop, popt_events, x_lim_events_norm = ff.hist_erlang_fit(events_norm, 'darkblue', label, ' (mod/ref)', '', ax)
    events_norm_drop, popt_events, x_lim_events_norm = ff.hist_erlang_fit(events_norm, 'darkblue', label, ' (mod/ref)', '', ax_all)
    ax.set_xlim(0,x_lim_events_norm)
    ax.set_xlabel('individually mormalized events [x-fold]', size=12)
    ax_all.set_xlim(0,x_lim_events_norm)
    ax_all.set_xlabel('individually mormalized events [x-fold]', size=12)
    
    f.savefig(save_path + '_normalized_events')
    f_all.savefig(save_path + '_all_kinetic_fits')
    
    norm_events_dict = {'n_events-sample_size'        : len(events_norm),           
                        'n_events-sample_size_erlang' : len(events_norm_drop),
      
                        'n_events_mod-mean'           : data_sec.mod_n_events.mean(),
                        'n_events_mod-median'         : data_sec.mod_n_events.median(),
                        'n_events_mod-std'            : data_sec.mod_n_events.std(),
                        'n_events_ref-mean'           : data_sec.ref_n_events.mean(),
                        'n_events_ref-median'         : data_sec.ref_n_events.median(),
                        'n_events_ref-std'            : data_sec.ref_n_events.std(),
                        
                        'n_events_norm-mean'          : events_norm.mean(),
                        'n_events_norm-median'        : events_norm.median(),
                        'n_events_norm-std'           : events_norm.std(),
                        'n_events_norm-mean_erlang'   : sp.stats.erlang.mean(*popt_events),
                        'n_events_norm-median_erlang' : sp.stats.erlang.median(*popt_events),
                        'n_events_norm-std_erlang'    : sp.stats.erlang.std(*popt_events)
                        }
    
    value_dict.update(norm_events_dict)
    return value_dict


