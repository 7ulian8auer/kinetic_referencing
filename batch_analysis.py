# Script to process _picked_props.hdf5 data
###################################################### Load packages
### platform independent paths
import os 
import importlib
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

### Load an extra custom file
import sys
sys.path.insert(1, 'C:/Julian_Bauer/lbFCS/MP')
'''sort by bright time'''
import plot_process_functions as ppf
'''sort by dark time'''
# import plot_process_functions_dark as ppf
import kr_io

###################################################### 
''' Defines path to data and sets input parameters'''
###################################################### Define data
dir_names=[]
dir_names.extend(['folder'])

file_names=[]
file_names.extend(['file....picked_props.hdf5'])

###################################################### Set parameters
#### temperatures optional
temperatures=[]
#### labels obligatory
labels=[]
'''Automatic label generation'''
# if len(labels) == 0:
#     labels=[]
#     for i in range(0,len(dir_names)):
#         ### Get label name out of the dir or file name
#         labels.append(dir_names[i].split('/')[-1][0:4])
#         labels.append(file_names[i][:4])


#%%
########################################## 
''' Read locs, apply props & save locs '''
##########################################
#### Create list of paths
path=[os.path.join(dir_names[i],file_names[i]) for i in range(0,len(file_names))]
tag_list=[]

#### Read-Apply-Save loop
for i in range(0,len(path)):
    print('File read in ...')
    props,props_info=lbfcs.io.load_locs(path[i])
    
    tag = file_names[i][:4]
    tag_list.append(tag)
    save_path = dir_names[i] + '/' + tag
    
    frames    = props_info[0]['Frames']
    CycleTime = float(props_info[0]['Micro-Manager Metadata']['Andor-ActualInterval-ms'])/1000
    label =labels[i]

    #################################
    print('1st filter (standard)...') 
    #################################
    '''Filter for DNA PAINT traces'''
    data_set = ppf.stat_filter(props,frames)
    
    '''plot filtered DATA'''
    ppf.stat_plot(props,data_set,frames,save_path)
    
    ###################################################
    print('Spot assignment ...') 
    ###################################################
    '''assign spots & drop single spots and clusters'''
    data_set_assigned, picks = ppf.spot_assignment(data_set)
    
    '''save assigned spots (pick per spot & pick per origami)'''
    ppf.picks_save(data_set_assigned,picks,path[i])
    
    ###########################################
    print('2nd filter (distance)...') 
    ###########################################
    '''plot spot distance & filter out 3xstd'''
    data_set_assigned_dist_filtered = ppf.distance_plot(data_set_assigned,label,save_path)
    
    '''save assigned picked-props'''
    lbfcs.io.save_locs(path[i].replace('.hdf5','_assigned.hdf5'),
                        data_set_assigned_dist_filtered,
                        props_info)
    
    ''' plot photon values'''
    ppf.photon_plot(data_set_assigned_dist_filtered,label,save_path)
    
    #########################################################################
    print('Generate DataFrame of kinetic values ...') 
    #########################################################################
    '''create characteristic values (modification, reference & normalized)'''
    data_sec_initial = data_set_assigned_dist_filtered.groupby('pair', as_index=False).apply(lambda df: ppf.bin_ds(df, sec=True, CycleTime=CycleTime))#, assign_crit='n_events'))
    
    ##################################################
    print('3rd filter (kinetic outlier)...') 
    ##################################################
    '''general outlier filtering by 3x gaussian std'''
    data_sec = ppf.outlier_plot(data_sec_initial, label, save_path)
    
    '''save kinetic-picked props'''
    lbfcs.io.save_locs(path[i].replace('.hdf5','_assigned_kin.hdf5'),
                        data_sec,
                        props_info)
    #########################################
    print('Generate dict of mean kinetic values ...') 
    #########################################
    '''generate dict of all kinetic values'''
    value_dict = ppf.kinetic_plots_values(data_sec, label, save_path)
    
    if len(temperatures)>0:
        additional_values={'CycleTime':     CycleTime,
                            'temperature':   temperatures[i], 
                            'modification':  tag
                            }
    else:
        additional_values={'CycleTime':     CycleTime,
                            'modification':  tag
                            }
    
    value_dict.update(additional_values)
    
    '''save single kinetic value dicts'''
    ppf.save_dict_to_file(value_dict, save_path)
    
    df_dict = pd.DataFrame.from_dict(value_dict, orient='index')
    df_dict_transpose = df_dict.T
    if i == 0:
        df_out = df_dict_transpose
    else:
        df_out = pd.concat([df_out,df_dict_transpose], ignore_index=True)
        

unique_tag_list = list(set(tag_list))
tag_names = '_'.join(unique_tag_list)
#### Save .hdf5 and .yaml of locs_props
print('File saving ...')
elements=path[0].split("/")
save_path_all = '/'.join(elements[0:-1]) 
lbfcs.io.save_locs(save_path_all + '/kinetic_values_' + tag_names + '.hdf5',
                    df_out,
                    props_info)


