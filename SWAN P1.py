#!/usr/bin/env python
# coding: utf-8

# In[6]:


from pandas import DataFrame
from datetime import datetime
from os import listdir
import numpy as np
import os
import pandas as pd
import csv
import numpy as np
import pickle
from tqdm import tqdm


# In[2]:


class MVTSSample:
    
    def __init__(self, active_region:str, flare_class:str, flare_type:str, \
                 verification:str, start_time:datetime, end_time:datetime, data:DataFrame):
        self.flare_type = flare_type
        self.active_region = active_region
        self.flare_class = flare_class
        self.verification = verification
        self.start_time = start_time
        self.end_time = end_time
        self.data = data
        
    def get_flare_type(self):
        return self.flare_type
    
    def get_flare_class(self):
        return self.flare_class
    
    def get_active_region(self):
        return self.active_region
    
    def get_verification(self):
        return self.verification
    
    def get_start_time(self):
        return self.start_time
    
    def get_end_time(self):
        return self.end_time
    
    def get_data(self):
        return self.data


# In[4]:


def read_mvts_instance(data_dir:str, file_name:str) -> MVTSSample:
    # Get flare type from file name
    if file_name[0:1] == 'F' :
        flare_type = file_name[0:2]
    else:
        flare_type = file_name[0:4]
    active_region = file_name[file_name.find('_ar')+3: file_name.find('_s2')]
    
    verification = 'FQ'
    flare_class = 'FQ'
    if file_name[0:1] != 'F' :
        verification = file_name.split(':')[1].split('_')[0]
        flare_class = file_name[0:1]
        
    try:
        # Get start time from file name
        start = file_name.find('s2')
        start_time = file_name[start+1: start+20]
        start_time = start_time.replace("T", " ")
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

        # Get end time from file name
        end = file_name.find('e2')
        end_time = file_name[end+1: end+20]
        end_time = end_time.replace("T", " ")
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print(ValueError)
        pass

    # Get data from csv file
    try:
        data = pd.read_csv(data_dir + "/" + file_name, sep="\t")
        data['Timestamp'] = data['Timestamp'].str.replace('-', '')
        data['Timestamp'] = data['Timestamp'].str.replace(' ', '')
        data['Timestamp'] = data['Timestamp'].str.replace(':', '')
    except ValueError:
        print(ValueError)
        pass
    
    # Make mvts object 
    mvts = MVTSSample(active_region, flare_class, flare_type, verification, start_time, end_time, data)
    return mvts


# In[7]:


def process_partition(partition_location:str, data_dir_save:str, abt_name:str):
    abt_header = ['Timestamp', 'R_VALUE','TOTUSJH','TOTBSQ','TOTPOT','TOTUSJZ','ABSNJZH','SAVNCPP',
                           'USFLUX','TOTFZ','MEANPOT', 'EPSX', 'EPSY','EPSZ','MEANSHR','SHRGT45','MEANGAM',
                              'MEANGBT','MEANGBZ','MEANGBH','MEANJZH','TOTFY','MEANJZD','MEANALP','TOTFX']
    
    abt_header_label = ['FLARE_CLASS', 'FLARE_TYPE', 'ACTIVE_REGION', 'VERIFICATION']
    
    abt_label = pd.DataFrame(columns=abt_header_label)
    

    # Get lists of data from partition
    FL = os.listdir(partition_location + "/FL")
    NF = os.listdir(partition_location + "/NF")
    FN_NO_FQ = [i for i in NF if i[0:2]!='FQ']
    FN_FQ = [i for i in NF if i[0:2]=='FQ']
    
    Files = sorted(FL, reverse=True) + sorted(FN_NO_FQ, reverse=True) + FN_FQ
    
    number_of_features=25
    number_of_timestamps=60
    abt = np.zeros((number_of_timestamps,number_of_features,len(Files)))
    
    count = 0
    # Add row to abt from mvssample object and its median and std data
    with tqdm(len(Files)) as pbar:

        for d in Files:

            # Use temp list for each row and temp df
            list2add_label = []
            tempdf = pd.DataFrame(columns=abt_header)
            tempdf_label = pd.DataFrame(columns=abt_header_label)

            # Get mvs object and add flare type 
            if d in FL:
                mvs = read_mvts_instance(partition_location + '/FL', d)
            else:
                mvs = read_mvts_instance(partition_location + '/NF', d)
            list2add_label.append(mvs.get_flare_class())
            list2add_label.append(mvs.get_flare_type())
            list2add_label.append(mvs.get_active_region())
            list2add_label.append(mvs.get_verification())


            # Set up temp df for future concat with master data frame object
            templist = mvs.get_data()[abt_header]
            templist = templist.to_numpy()

            # From data frame concat current with temp for each feature 
            abt[:,:,count] = templist

            tempdf_label.loc[len(abt_header_label)] = list2add_label
            abt_label = pd.concat([abt_label, tempdf_label], ignore_index= True, axis = 0)


            count +=1
            pbar.update(1)
            

    print(abt_name)        
    print("shape: " + str(abt.shape))
    with open(data_dir_save + "1_Raw/" + abt_name + ".pkl", 'wb') as f:
        pickle.dump(abt, f)
        
    abt_label.to_csv(data_dir_save + "2_Labels/" + abt_name + "_Labels.csv", index=False, header=True)
    # return the completed analitics base table
    return abt_label


# In[10]:


data_dir = "/Users/ankithreddy/Downloads/SWANraw/"  
data_dir_save = "/Users/ankithreddy/Downloads/SWANpre/"  

# change the path to where your data is stored.

num_partitions = 5

for i in range(0,num_partitions):
    abt_name = "partition" + str(i+1)
    abt = process_partition(data_dir + abt_name, data_dir_save, abt_name)
    print("number of instances: " + str(abt.shape[0]))
    print(abt.head(5))
    print('\n')


# In[11]:


# MISSING VALUE EXPLORATION


# In[14]:


# Loading the Raw Data
import pickle
import numpy as np

data_dir = "/Users/ankithreddy/Downloads/SWANpre/1_Raw/"
raw_data = []

num_partitions = 5

for i in range(0,num_partitions):
# Load the array with Pickle
    with open(data_dir + "partition" + str(i+1) + ".pkl", 'rb') as f:
        raw_data.append(pickle.load(f))


# In[16]:


def print_missing_values(data, start_partition, end_partition):
    abt_header = ['Timestamp', 'R_VALUE','TOTUSJH','TOTBSQ','TOTPOT','TOTUSJZ','ABSNJZH','SAVNCPP',
                               'USFLUX','TOTFZ','MEANPOT', 'EPSX', 'EPSY','EPSZ','MEANSHR','SHRGT45','MEANGAM',
                                  'MEANGBT','MEANGBZ','MEANGBH','MEANJZH','TOTFY','MEANJZD','MEANALP','TOTFX']
    num_columns = 25
    num_timestamps = 60
    num_partitions = 5
    null_count = [0,0,0,0,0]
    non_null_count = [0,0,0,0,0]
    null_count_per_feature = np.zeros((num_partitions,num_columns), dtype=int)

    for i in range(start_partition-1, end_partition):
        partition = np.array(data[i])

        for j in range(0,partition.shape[2]):
            mvts = partition[:,:, j]
            for m in range(0,num_columns):
                for n in range (0,num_timestamps):
                    if (mvts[n,m] == 0.0 or mvts[n,m] == np.nan):
                        null_count[i] += 1
                        null_count_per_feature[i,m] += 1
                    else:
                        non_null_count[i] += 1

        print("Partition" + str(i+1) + ":\n")
        print("null counts in P" + str(i+1) + ": " + str(null_count[i]))
        print("non-null counts in P"+ str(i+1) + ": " + str(non_null_count[i]))
        for x in range(0,num_columns):
            print(abt_header[x] + ": " + str(null_count_per_feature[i,x]))

        print("\n")


# In[17]:


print_missing_values(raw_data,1,5)


# In[ ]:




