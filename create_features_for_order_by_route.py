import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import logging
from math import radians, cos, sin, asin, sqrt

logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)

def geodistance(x,lng1,lat1,lng2,lat2):
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(x[lng1]), float(x[lat1]), float(x[lng2]), float(x[lat2])]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km # unit: m
    distance=round(distance/1000,3) # unit: km
    return distance



def split_ports(x,i):
    try:
        res=re.split(r'[>-]',x)[i]
        return res
    except Exception as e:
        return ""
    
def preprocess_order(df_order):
    df_order = df_order[~df_order.timestamp.isna()].copy()
    df_order.timestamp = pd.to_datetime(df_order.timestamp)
    df_order.sort_values(by="timestamp",inplace=True)
    df_order["start_port"]=df_order["vesselNextport"].apply(lambda x:split_ports(x,0))   
    df_order["end_port"]=df_order["vesselNextport"].apply(lambda x:split_ports(x,-1))
    return df_order

def create_schedule(df_processed):
    df_start_min = df_processed[df_processed.start_port!=""].groupby("start_port").agg({"timestamp":np.min}).sort_values("timestamp")
    df_start_max = df_processed[df_processed.start_port!=""].groupby("start_port").agg({"timestamp":np.max}).sort_values("timestamp")
    df_route = pd.concat([df_start_min,df_start_max],axis=1)
    df_route.columns = ["start_timestamp","end_timestamp"]
    df_route.sort_values(by="start_timestamp",inplace=True)
    
    # 清洗route数据表
    last_endstamp = None
    for row_id,row in df_route.iterrows():
        if last_endstamp is None:
            last_endstamp=row["end_timestamp"]
        else:
            if row["end_timestamp"]<last_endstamp:
                df_route.drop(row_id,inplace=True)
            else:
                last_endstamp = row["end_timestamp"]
    diff_time = (df_route["end_timestamp"]-df_route["start_timestamp"]).dt.total_seconds()/3600
    return df_route[diff_time>2]

def time_to_location(df_route,df_order):
    df_time_location = pd.DataFrame(columns=["start_timestamp","start_longitude","start_latitude"
                                            ,"end_timestamp","end_longitude","end_latitude","number"])
    idx = 0
    df_order.timestamp = pd.to_datetime(df_order.timestamp)
    for i, row in df_route.iterrows():
        start_t = row["start_timestamp"]
        df_part = df_order[(df_order.timestamp>(start_t-pd.Timedelta(10,unit="m")))&
         (df_order.timestamp<(start_t+pd.Timedelta(10,unit="m")))]  
        start_long = df_part["longitude"].median()
        start_lat = df_part["latitude"].median()
        
        if idx > 0:
            df_time_location.loc[idx]=[end_t,end_long,end_lat,start_t,start_long,start_lat,idx]
            idx+=1
        
        end_t = row["end_timestamp"]
        df_part = df_order[(df_order.timestamp>(end_t-pd.Timedelta(10,unit="m")))&
         (df_order.timestamp<(end_t+pd.Timedelta(10,unit="m")))]  
        end_long = df_part["longitude"].median()
        end_lat = df_part["latitude"].median()
        df_time_location.loc[idx]=[start_t,start_long,start_lat,end_t,end_long,end_lat,idx]
        idx+=1
    return df_time_location

def order_to_schedule(read_path, write_path,route_path):
    files = os.listdir(read_path)
    
    for f in files:
        if os.path.exists(write_path+f):
            continue
        logging.info("Creating schedule for the order {}".format(f))
        df_order=pd.read_csv(read_path+f,index_col=0)
        df_processed = preprocess_order(df_order)
        df_route = create_schedule(df_processed)
        df_route.to_csv(route_path+f)
        df_schedule = time_to_location(df_route,df_order)
        df_schedule.to_csv(write_path+f)

def encode_feature_single_order(df_schedule, df_order):
    nrow= df_order.shape[0]
    orders = [0]*nrow
    i=0
    
    for idx, row in df_order.iterrows():
        aim_schedule = df_schedule[df_schedule.start_timestamp<=pd.to_datetime(row["timestamp"])][-1:]
        try:
            aim_number = aim_schedule["number"].values[0]
        except:
            #print("Something is wrong",aim_schedule["number"])
            aim_number = -1
        orders[i]=aim_number
        i+=1
    df_order["number"]=orders
    df_features = pd.merge(df_order[df_order.number>=0],df_schedule,on="number",how="left")
    df_features["long_diff_from_start"]=df_features["longitude"]-df_features["start_longitude"]
    df_features["lat_diff_from_start"]=df_features["latitude"]-df_features["start_latitude"]
    df_features["long_diff_to_end"]=df_features["end_longitude"]-df_features["longitude"]
    df_features["lat_diff_to_end"]=df_features["end_latitude"]-df_features["latitude"]
    df_features["distance_from_start"]=df_features.apply(lambda x:geodistance(x,"longitude","latitude","start_longitude","start_latitude"),axis=1)
    df_features["distance_to_end"]=df_features.apply(lambda x:geodistance(x,"longitude","latitude","end_longitude","end_latitude"),axis=1)
    
    df_features["time_used"]=pd.to_datetime(df_features["timestamp"])-pd.to_datetime(df_features["start_timestamp"])
    df_features["time_used"]=df_features["time_used"].dt.total_seconds()
    df_features["label"]=pd.to_datetime(df_features["end_timestamp"])-pd.to_datetime(df_features["timestamp"])
    df_features["label"]=df_features["label"].dt.total_seconds()
    col_features = ["loadingOrder","long_diff_from_start","lat_diff_from_start","long_diff_to_end",
                "lat_diff_to_end","start_longitude","start_latitude","end_longitude","end_latitude",
                    "distance_from_start","distance_to_end",
                "speed","direction","time_used","label"]
    return df_features[col_features]

#%%

if __name__ == "__main__":
    base_directory = "F:/grad/HWC/ChuSai/"
    test = pd.read_csv(base_directory+"data/B_testData0626.csv")
    routes = test["TRANSPORT_TRACE"].unique()
    
# setting to your own path
    i = 9
    order_path = base_directory+"data/order/"+routes[i]+"/"
    schedule_path = base_directory+"data/schedule/"+routes[i]+"/"
    feature_path = base_directory+"data/feature/"+routes[i]+"/"
    route_path = base_directory+"data/route/"+routes[i]+"/"
    

    if not os.path.exists(schedule_path):
    	os.mkdir(schedule_path)
    if not os.path.exists(route_path):
    	os.mkdir(route_path)

    order_to_schedule(order_path,schedule_path,route_path)

    if not os.path.exists(feature_path):
    	os.mkdir(feature_path)

    files = os.listdir(order_path)

    for f in files:
    	if os.path.exists(feature_path+f):
    		continue

    	logging.info("Creating features for order:{}".format(f))
    	df_order = pd.read_csv(order_path+f,index_col=0)
    	df_schedule = pd.read_csv(schedule_path+f,index_col=0)
    	if df_schedule.shape[0]==0:
            continue
        	
    	df_order.timestamp=pd.to_datetime(df_order.timestamp)
    	df_schedule.start_timestamp = pd.to_datetime(df_schedule.start_timestamp)
    	df_schedule.end_timestamp = pd.to_datetime(df_schedule.end_timestamp)
    
    	df_feature = encode_feature_single_order(df_schedule,df_order)
    	df_feature.to_csv(feature_path+f)
        
#%%
df_test = pd.read_csv('F:/grad/HWC/ChuSai/data/B_testData0626.csv')
def encode_feature_for_test(df_test):
    df_port = pd.read_csv('F:/grad/HWC/ChuSai/data/port_fixed_0612.csv')
    orders = df_test['loadingOrder'].unique()

    col_features = ["loadingOrder","long_diff_from_start","lat_diff_from_start","long_diff_to_end",
            "lat_diff_to_end","start_longitude","start_latitude","end_longitude","end_latitude",
                "distance_from_start","distance_to_end",
            "speed","direction","time_used"]
    df_tf = pd.DataFrame(columns=col_features)
    for i in range(len(orders)):
        print('i = ',i)
        df_order = df_test[df_test['loadingOrder']==orders[i]]
        fromto = df_order['TRANSPORT_TRACE'].unique()[0].split('-')
        df_target0 = df_port[df_port['TRANS_NODE_NAME'] == fromto[0]]
        df_target1 = df_port[df_port['TRANS_NODE_NAME'] == fromto[-1]]
        print(df_target1.shape)
        nrow= df_order.shape[0]
        df_features = pd.DataFrame()
        df_features['loadingOrder'] = df_order['loadingOrder']
        df_features["long_diff_from_start"]=df_order["longitude"]-df_target0['LONGITUDE'].iloc[-1]
        df_features["lat_diff_from_start"]=df_order["latitude"]-df_target0['LATITUDE'].iloc[-1]
        df_features["long_diff_to_end"]=df_target1['LONGITUDE'].iloc[-1]-df_order["longitude"]
        df_features["lat_diff_to_end"]=df_target1['LATITUDE'].iloc[-1]-df_order["latitude"]
        
        df_features["longitude"]=df_order["longitude"]
        df_features["latitude"]=df_order["latitude"]
        
        df_features['start_longitude'] = np.repeat(df_target0['LONGITUDE'].iloc[-1],nrow)
        df_features['start_latitude'] = np.repeat(df_target0['LATITUDE'].iloc[-1],nrow)
        df_features['end_longitude'] = np.repeat(df_target1['LONGITUDE'].iloc[-1],nrow)
        df_features['end_latitude'] = np.repeat(df_target1['LATITUDE'].iloc[-1],nrow)
        
        df_features["distance_from_start"]=df_features.apply(lambda x:geodistance(x,"longitude","latitude","start_longitude","start_latitude"),axis=1)
        df_features["distance_to_end"]=df_features.apply(lambda x:geodistance(x,"longitude","latitude","end_longitude","end_latitude"),axis=1)
        
        df_features['speed']=df_order['speed']
        df_features['direction']=df_order['direction']
        df_features["time_used"]=pd.to_datetime(df_order["timestamp"])-pd.to_datetime(list(df_order["onboardDate"]),utc=True)
        df_features["time_used"]=df_features["time_used"].dt.total_seconds()
        df_features = df_features[col_features]

        df_tf = pd.concat([df_tf,df_features])
    print('the size of dt_tf is ',df_tf.shape)
    # df_tf.to_csv('F:/grad/HWC/ChuSai/data/test_featureB.csv',index=False)
    return df_features[col_features]


#%%
encode_feature_for_test(df_test)