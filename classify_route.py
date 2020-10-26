import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import logging
from math import radians, cos, sin, asin, sqrt

logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%y-%m-%d %H:%M:%S', level=logging.INFO)

#%%
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

def cut_route(schedule_path,order_name,aim_start_long,aim_start_lat,aim_end_long,aim_end_lat,diff_allowed):
    start_row=None
    end_row=None
    df_schdeule = pd.read_csv(schedule_path+order_name+".csv",index_col=0)
    if df_schdeule.shape[0]<1:
        return None
    df_schdeule["distance_start_end"]=df_schdeule.apply(lambda x:geodistance(x,"start_longitude","start_latitude","end_longitude","end_latitude"),axis=1)
    for row_id, row in df_schdeule.iterrows():
        print(row)
        if abs(row["start_longitude"]-aim_start_long)<diff_allowed and abs(row["start_latitude"]-aim_start_lat)<diff_allowed:
            start_row = row_id
            #print(start_row)
        if abs(row["end_longitude"]-aim_end_long)<diff_allowed and abs(row["end_latitude"]-aim_end_lat)<diff_allowed:
            if end_row is None:
                end_row = row_id
                #print(end_row)
    if start_row is not None and end_row is not None and start_row<=end_row:
        df_short = df_schdeule.loc[start_row:end_row]
        print("Finish cutting the route, retrieving the information for the order: ",order_name,".")
        return df_short
    return None
            

def generate_order_summary(df,order_name,partial_route):
    df["distance_start_end"]=df.apply(lambda x: geodistance(x,"start_longitude","start_latitude","end_longitude","end_latitude"),axis=1)
    df["time_used"]=(pd.to_datetime(df["end_timestamp"])-pd.to_datetime(df["start_timestamp"])).dt.total_seconds()/3600
    df["speed"]=df["distance_start_end"]/df["time_used"]
    # df["time_still"]=df.apply(lambda x: x["time_used"] if x["speed"]<1 else 0,axis=1)
    df["time_used"]=df["time_used"].cumsum()
    # df["time_still"]=df["time_still"].cumsum()
    # df["time_moving"]=df["time_used"]-df["time_still"]
    
    df_report=pd.DataFrame(columns=["loaingOrder","start_timestamp","start_longitude","start_latitude","end_timestamp","end_longitude","end_latitude","TRANSPORT_TRACE"])
    df_report.loc[0]=[order_name,df["start_timestamp"].values[0],df["start_longitude"].values[0],df["start_latitude"].values[0],
                     df["end_timestamp"].values[-1],df["end_longitude"].values[-1],df["end_latitude"].values[-1],partial_route]
    return df_report
                     

def generate_route_set_info(route_name, route_set_path, schedule_path, df_port_short):
    aim_route_info = route_name.split("-")
    terminal_long = []
    terminal_lat = []
    port_cluster = []
    for i in range(len(aim_route_info)):
        terminal = df_port_short[df_port_short.TRANS_NODE_NAME==aim_route_info[i]]
        terminal_long.append(terminal["LONGITUDE"].values[0])
        terminal_lat.append(terminal["LATITUDE"].values[0])
        port_cluster.append(terminal["cluster"].values[0])
    
    for i in range(len(aim_route_info)-1):
        start_long = terminal_long[i]
        start_lat = terminal_lat[i]
        end_long = terminal_long[i+1]
        end_lat = terminal_lat[i+1]
        partial_route = "-".join(aim_route_info[i:i+2])
        
        print("Searching Schedule Files, finding",i+1,"-th possible combinations of the route: ",partial_route,".")
        schedule_files = os.listdir(schedule_path)
        df_route_set= None
        if port_cluster[i] == port_cluster[i+1]:
            diff_allowed = 0.2
        else:
            diff_allowed = 2
        for s_file in sorted(schedule_files):
            order_name = s_file[:-4]
            df_cut= cut_route(schedule_path,order_name,start_long,start_lat,end_long,end_lat,diff_allowed)
            if df_cut is not None:
                df_summary = generate_order_summary(df_cut,order_name,partial_route)
                df_route_set=pd.concat([df_route_set,df_summary])
    if df_route_set is not None:
        df_route_set.to_csv(route_set_path+route_name+".csv",index=False)
        
#%%
if __name__ == "__main__":
    # setting to your own path
    basic_path = 'F:/grad/HWC/FuSai/'
    order_path = basic_path+"data/order/"
    schedule_path = basic_path+"data/schedule/"
    feature_path = basic_path+"data/feature/"
    route_path = basic_path+"data/route/"
    route_set_path = basic_path + "data/route_set/"
    
    columns_list = ['loadingOrder','timestamp','longitude','latitude','speed','direction','vesselNextport','TRANSPORT_TRACE']
    test = pd.read_csv(basic_path+"data/R2_ATest0711.csv")
    route_name = sorted(test["TRANSPORT_TRACE"].unique())
    df_port_short = pd.read_csv(basic_path + "data/port_fixed_0612.csv")
    
    # df_port_short = df_port_short[df_port_short['TRANS_NODE_NAME'].isin(port_list)]
    # port_data = port_data.drop_duplicates(subset = ['TRANSPORT_NODE_ID'],keep = 'first')
    # port_data = port_data.drop_duplicates(subset = ['TRANS_NODE_NAME'],keep = 'first')
    
    # group = 9
    # for i in range(group*10,min(len(route_name),(group+1)*10)):
    i = 59
    generate_route_set_info(route_name[i], route_set_path, schedule_path, df_port_short)
    