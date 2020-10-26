import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import logging
from math import radians, cos, sin, asin, sqrt

logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)

class TimeRecord():
    def __init__(self,*lst):
        assert len(lst)==8
        self._min_id= lst[0]
        self._min_time = pd.to_datetime(lst[1])
        self._min_long = lst[2]
        self._min_lat = lst[3]
        self._max_id = lst[4]
        self._max_time = pd.to_datetime(lst[5])
        self._max_long = lst[6]
        self._max_lat = lst[7]
        
    def update(self,*lst):
        assert len(lst)==8
        cur_min = pd.to_datetime(lst[1])
        if cur_min<self._min_time:
            self._min_time = cur_min
            self._min_id = lst[0]
            self._min_long = lst[2]
            self._min_lat = lst[3]
        cur_max = pd.to_datetime(lst[5])
        if cur_max>self._max_time:
            self._max_time = cur_max
            self._max_id = lst[4]
            self._max_long = lst[6]
            self._max_lat = lst[7]
            
    def time_diff(self):
        return self._max_time- self._min_time
    
    def get_min(self):
        return self._min_id, self._min_time, self._min_long,self._min_lat
    
    def get_max(self):
        return self._max_id, self._max_time,self._max_long, self._max_lat
            
    def __str__(self):
        return "({},{},{},{},{},{},{},{})".format(self._min_id,self._min_time,self._min_long,self._min_lat,
                                                  self._max_id,self._max_time,self._max_long, self._max_lat)
    def __repr__(self):
        return "({},{},{},{},{},{},{},{})".format(self._min_id,self._min_time,self._min_long,self._min_lat,
                                                  self._max_id,self._max_time,self._max_long, self._max_lat)
    
def extract_min_max(x):
    min_id = x["timestamp"].idxmin()
    max_id = x["timestamp"].idxmax()
    return [min_id, x["timestamp"].loc[min_id],x["longitude"].loc[min_id],x["latitude"].loc[min_id],
            max_id,x["timestamp"].loc[max_id],x["longitude"].loc[max_id],x["latitude"].loc[max_id]]

def summary_order_info(read_path,routes):
    col_names = ['loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']
    df = pd.read_csv(read_path+routes+".csv", header =None, names = col_names, iterator=True)
    load_dict=dict()
    chunk_cnt= 0
    #start_time = datetime.now()
    logging.info("Extracting order's earliest time and lastest time information...")
    while True:
        try:
            df_train = df.get_chunk(1000000)
            df_train.timestamp = pd.to_datetime(df_train.timestamp)
            df_time_diff = df_train.groupby("loadingOrder").apply(lambda x: extract_min_max(x))
            for load,rec in zip(df_time_diff.index,df_time_diff.values):
                if load in load_dict.keys():
                    load_dict[load].update(*rec)
                else:
                    load_dict[load]=TimeRecord(*rec)
            print("The {} chunk, with length {}".format(chunk_cnt+1,len(load_dict.keys())))
            chunk_cnt+=1
        except:
            break
    #logging.info("Time used: ",datetime.now()-start_time)
    return load_dict

def select_orders(load_dict):
    logging.info("Transforming order info file...")
    #start_time = datetime.now()
    for key,val in load_dict.items():
        min_infos = val.get_min()
        max_infos = val.get_max()
        load_dict[key]=[min_infos[0],min_infos[1],min_infos[2],min_infos[3],
                       max_infos[0],max_infos[1],max_infos[2],max_infos[3]]
    df_min_max = pd.DataFrame.from_dict(load_dict,orient="index",
                                    columns=['Min_Time_RowID', 'Min_Time','Min_Long','Min_Lat',
                                             'Max_Time_RowID', 'Max_Time','Max_Long','Max_Lat'])
    df_min_max["loadingOrder"]=df_min_max.index
    #logging.info("Time used: ",datetime.now()-start_time)
    return df_min_max


def geodistance(x,lng1,lat1,lng2,lat2):
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(x[lng1]), float(x[lat1]), float(x[lng2]), float(x[lat2])]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km # unit: m
    distance=round(distance/1000,3) # unit: km
    return distance



def clean_strange_order_data(df_order):
    cols = df_order.columns
    df_order.timestamp = pd.to_datetime(df_order.timestamp)
    df_order.sort_values(by="timestamp",inplace=True)
    df_order["diff_seconds"]=df_order['timestamp'].diff(1).dt.total_seconds()
    df_after = df_order[1:].copy()#只保留第一行之后的数据
    df_after["long_before"]=df_order["longitude"][0:-1].values
    df_after["lat_before"]=df_order["latitude"][0:-1].values
    df_after["distance_diff"]=df_after.apply(lambda x:geodistance(x,"longitude","latitude","long_before","lat_before"),axis=1)
    df_after["real_speed"]=df_after["distance_diff"]*3600/df_after["diff_seconds"]
    df_good = df_after[df_after.real_speed<50].copy()
    return df_good[cols]

def save_order(order_record,route,save_path):
    order_name=order_record["loadingOrder"]
    if os.path.exists(save_path+order_name+".csv"):
    	logging.info("Order: {} has already existed.".format(order_name))
    	return None

    min_id = order_record["Min_Time_RowID"]#.values[0]
    max_id = order_record["Max_Time_RowID"]#.values[0]
    col_names = ['loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']

    df=pd.read_csv("F:/grad/HWC/ChuSai/data/train/"+route+".csv",header =None, names = col_names,chunksize=1000000)
    logging.info("Loaded Big File...")
    df_order = None
    #i = 0
    logging.info("Order: {}. The first row id: {}, The last row id: {}".format(order_name, min_id, max_id))
    read_sign = False
    for chunk in df:
        chunk_min = chunk.index.min()
        chunk_max = chunk.index.max()
        #print("The chunk min id: {}, The chunk max id: {}".format(chunk_min,chunk_max))
        if read_sign:
            df_aim = chunk[chunk.loadingOrder==order_name]
            if df_aim.shape[0]>0:
                df_order = pd.concat([df_order,df_aim])
        elif chunk_max>=min_id:
            read_sign=True
            logging.info("Reading chunk...")
            df_aim = chunk[chunk.loadingOrder==order_name]
            if df_aim.shape[0]>0:
                df_order = pd.concat([df_order,df_aim])
        if chunk_min>=max_id:
            break
        
        #i+=1
    if df_order is not None:
    	df_order = clean_strange_order_data(df_order)
    	logging.info("Saving the order file.")
    	df_order.to_csv(save_path+order_name+".csv")

#%%
if __name__ == "__main__":
    base_directory = "F:/grad/HWC/ChuSai/"
    test = pd.read_csv(base_directory+"data/B_testData0626.csv")
    routes = test["TRANSPORT_TRACE"].unique()
    
    i = 10
    # setting to your own path
    cargo_data_path = "F:/grad/HWC/Chusai/data/" # where the train0622.csv file is stored.
    min_max_file_path = "F:/grad/HWC/Chusai/data/all_routes/"
    order_path = "F:/grad/HWC/Chusai/data/order/"+routes[i]+'/'
    read_path = "F:/grad/HWC/Chusai/data/train/"
    if not os.path.exists(order_path):
    	os.mkdir(order_path)
    
    if not os.path.exists(min_max_file_path+routes[i]+".csv"):
    	load_dict = summary_order_info(read_path,routes[i])
    	df_min_max = select_orders(load_dict)
    	df_min_max.to_csv(min_max_file_path+routes[i]+".csv")
    else:
    	df_min_max = pd.read_csv(min_max_file_path+routes[i]+".csv",index_col=0)
    
    df_min_max["day_diff"]=(pd.to_datetime(df_min_max.Max_Time)-\
                            pd.to_datetime(df_min_max.Min_Time)).dt.total_seconds()/3600/24
    df_min_max_sub = df_min_max[df_min_max.day_diff>4]
    
    # create your testing dataset
    df_sample = df_min_max_sub
    
    for idx,order in df_sample.iterrows():
        save_order(order,routes[i],order_path)
            
#%%
def extract_min_max(x):
    min_id = x["timestamp"].idxmin()
    max_id = x["timestamp"].idxmax()
    return [min_id, x["timestamp"].loc[min_id],x["longitude"].loc[min_id],x["latitude"].loc[min_id],
            max_id,x["timestamp"].loc[max_id],x["longitude"].loc[max_id],x["latitude"].loc[max_id]]

def test_order_info():
    col_names = ['loadingOrder','timestamp','longitude',
                  'latitude','speed','direction','vesselMMSI','carrierName'\
                      'onboardDate','TRANSPORT_TRACE']
    df = pd.read_csv("F:/grad/HWC/ChuSai/data/B_testData0626.csv")
    df.index = np.arange(df.shape[0])
    load_dict=dict()
    chunk_cnt= 0
    #start_time = datetime.now()
    logging.info("Extracting order's earliest time and lastest time information...")
    df.timestamp = pd.to_datetime(df.timestamp)
    df_time_diff = df.groupby("loadingOrder").apply(lambda x: extract_min_max(x))
    for load,rec in zip(df_time_diff.index,df_time_diff.values):
        if load in load_dict.keys():
            load_dict[load].update(*rec)
        else:
            load_dict[load]=TimeRecord(*rec)
    #logging.info("Time used: ",datetime.now()-start_time)
    return load_dict

def select_orders(load_dict):
    logging.info("Transforming order info file...")
    #start_time = datetime.now()
    for key,val in load_dict.items():
        min_infos = val.get_min()
        max_infos = val.get_max()
        load_dict[key]=[min_infos[0],min_infos[1],min_infos[2],min_infos[3],
                       max_infos[0],max_infos[1],max_infos[2],max_infos[3]]
    df_min_max = pd.DataFrame.from_dict(load_dict,orient="index",
                                    columns=['Min_Time_RowID', 'Min_Time','Min_Long','Min_Lat',
                                             'Max_Time_RowID', 'Max_Time','Max_Long','Max_Lat'])
    df_min_max["loadingOrder"]=df_min_max.index
    #logging.info("Time used: ",datetime.now()-start_time)
    return df_min_max

load_dict=test_order_info()
df_min_max =select_orders(load_dict)
df_min_max.to_csv('F:/grad/HWC/ChuSai/data/test_routes.csv',index=False)