import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
from copy import deepcopy
import random
from annotation import annotation, get_labels
import threading
import multiprocessing
sys.path.append("../../")
from configs import *
from utils import *
random.seed(0)

def process_A(n_threads=N_CORES):
    raw_single_root = ROOT+"/data/data_collection/single/"
    label_single_root = ROOT+"/data/data_process/labels_single/" 
    raw_multiple_root = ROOT+"/data/data_collection/multiple/"
    label_multiple_root = ROOT+"/data/data_process/labels_multiple/" 
    data_root = ROOT+"/data/data_process/Model_A/"
    tmp_root = ROOT+"/data/data_process/Model_A/tmp/"

    mutex=threading.Lock()
    paths=[]
    parallel_dict={"loop_cnts":0,"last_timestamp":None}

    for path_name in walk(raw_multiple_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                paths.append(path_RPS)

    def t_process_A(paths,idx_start,idx_end,parallel_dict):
        for idx in range(idx_start,idx_end):
            if idx >= len(paths):
                continue

            mutex.acquire()
            parallel_dict["loop_cnts"]+=1
            if parallel_dict["loop_cnts"]%10==0:
                percentage=round(parallel_dict["loop_cnts"]/len(paths),4)*100
                if parallel_dict["last_timestamp"] is not None:
                    speed=round(1000/(time.time()-parallel_dict["last_timestamp"]),2)
                    ETA=round(len(paths)/speed,2)
                    print("Loops:{}/{}, {}%, speed:{}, ETA:{}".format(parallel_dict["loop_cnts"],len(paths),percentage,speed,ETA))
                parallel_dict["last_timestamp"]=time.time()
            mutex.release()

            path_RPS=paths[idx]
            colocation, name, n_thread, RPS = path_RPS.split('/')[-4:]
            tmp_dir = tmp_root+"{}/{}/{}/".format(colocation,name,n_thread)
            tmp_path = tmp_dir+"{}.csv".format(RPS)

            if os.path.exists(tmp_path):
                continue
            labels = get_labels(path_RPS, label_multiple_root)
            if labels is None:
                print(path_RPS+"label is none")
                continue
            else:
                path_read = path_RPS + "/{}+{}+{}+{}+{}".format(name, n_thread, RPS, labels["OAA_Core"], cache_2_way(labels["OAA_Cache"],MB_PER_WAY))
                if not os.path.exists(path_read):
                    continue
                OAA_df = pd.read_csv(path_read)
                #OAA_bandwidth = OAA_df["MBL"].mean()
                #labels["OAA_Bandwidth"] = OAA_bandwidth

            RPS_df=pd.DataFrame()
            for path_case in walk(path_RPS):
                try:
                    df = pd.read_csv(path_case)
                    df = df[A_FEATURES]
                except pd.errors.EmptyDataError as e:
                    print(path_case, ' is empty')
                    os.system('rm -f {}'.format(path_case))
                RPS_df = pd.concat([RPS_df,df], ignore_index=True, sort=False)
            
            for key in labels:
                RPS_df[key] = labels[key]
            RPS_df=RPS_df[A_FEATURES+A_LABELS]
            os.system("mkdir -p {}".format(tmp_dir))
            RPS_df.to_csv(tmp_path, index=None)
    
    threads=[]
    blk_length=len(paths)//n_threads+1
    for t_idx in range(n_threads):
        idx_start=t_idx*blk_length
        idx_end=(t_idx+1)*blk_length
        thread=threading.Thread(target=t_process_A,args=(paths,idx_start,idx_end,parallel_dict))
        threads.append(thread)
    for t in threads:
        t.start() 

def process_B(n_threads=N_CORES):
    raw_single_root = ROOT+"/data/data_collection/single/"
    raw_multiple_root = ROOT+"/data/data_collection/multiple/"
    data_root = ROOT+"/data/data_process/Model_B/"
    tmp_root = ROOT+"/data/data_process/Model_B/tmp/" 
    label_root = ROOT+"/data/data_process/labels_multiple/"

    mutex=threading.Lock()
    paths=[]
    parallel_dict={"loop_cnts":0,"last_timestamp":None}
    for path_name in walk(raw_multiple_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                paths.append(path_RPS)

    def t_process_B(paths,idx_start,idx_end,parallel_dict):
        for idx in range(idx_start,idx_end):
            if idx >= len(paths):
                continue

            mutex.acquire()
            parallel_dict["loop_cnts"]+=1
            if parallel_dict["loop_cnts"]%10==0:
                percentage=round(parallel_dict["loop_cnts"]/len(paths),4)*100
                if parallel_dict["last_timestamp"] is not None:
                    speed=round(1000/(time.time()-parallel_dict["last_timestamp"]),2)
                    ETA=round(len(paths)/speed,2)
                    print("Loops:{}/{}, {}%, speed:{}, ETA:{}".format(parallel_dict["loop_cnts"],len(paths),round(parallel_dict["loop_cnts"]/len(paths),4)*100,speed,ETA))
                parallel_dict["last_timestamp"]=time.time()
            mutex.release()

            path_RPS = paths[idx]
            colocation, name, n_thread, RPS = path_RPS.split('/')[-4:]
            tmp_dir = tmp_root+"{}/{}/{}/".format(colocation,name,n_thread)
            tmp_path = tmp_dir+"{}.csv".format(RPS)

            RPS_df=pd.DataFrame()
            if os.path.exists(tmp_path):
                continue
            
            all_allos=[]
            lats = {}
            dfs = {}
            for core_idx in range(N_CORES):
                for way_idx in range(N_WAYS):
                    path_case = path_RPS + "/" + "{}+{}+{}+{}+{}".format(name, n_thread, RPS, core_idx, way_idx)
                    if not os.path.exists(path_case):
                        continue
                    df = pd.read_csv(path_case)
                    avg_latency = df['Latency'].mean()
                    lats[(core_idx, way_idx)] = avg_latency
                    dfs[(core_idx, way_idx)] = df
                    all_allos.append((core_idx,way_idx))


            for core_idx in range(N_CORES):
                for way_idx in range(N_WAYS):
                    path_case = path_RPS + "/" + "{}+{}+{}+{}+{}".format(name, n_thread, RPS, core_idx, way_idx)
                    if not os.path.exists(path_case):
                        continue
                    df = pd.read_csv(path_case)
                    for i in range(df.shape[0]):
                        to_allo=random.sample(all_allos, 1)[0]
                        df.loc[i,"Target_Cache"]=way_2_cache(to_allo[1],MB_PER_WAY)
                        df.loc[i,"Target_Core"]=to_allo[0]
                        df.loc[i,"QoS"]=QOS_TARGET[name]/lats[to_allo]

                    RPS_df = pd.concat([RPS_df,df], ignore_index=True, sort=False)
            print(RPS_df.columns)
            if RPS_df.shape[0]==0:
                continue
            RPS_df=RPS_df[B_FEATURES+B_LABELS]
            os.system("mkdir -p {}".format(tmp_dir))
            RPS_df.to_csv(tmp_path, index=None)

    threads=[]
    blk_length=len(paths)//n_threads+1
    for t_idx in range(n_threads):
        idx_start=t_idx*blk_length
        idx_end=(t_idx+1)*blk_length
        thread=threading.Thread(target=t_process_B,args=(paths,idx_start,idx_end,parallel_dict))
        threads.append(thread)
    for t in threads:
        t.start()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ["Model_A", "Model_B"]
    func_map = {"Model_A": process_A,"Model_B": process_B}
    for dataset in datasets:
        func_map[dataset]()
