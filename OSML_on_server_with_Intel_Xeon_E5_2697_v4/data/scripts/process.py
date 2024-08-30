import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from copy import deepcopy
from annotation import annotation, get_labels
import threading
import multiprocessing
sys.path.append("../../")
from utils import *
from configs import *

# Set N_CORES and N_WAYS the same as the platform used for collecting the data
N_CORES=36
CORE_INDEX = list(range(0, N_CORES))
N_WAYS=20
WAY_INDEX = list(range(0, N_WAYS))
MB_PER_WAY=2.25

def process_A(n_threads=N_CORES):
    raw_root = ROOT+"data/data_collection/single/"
    data_root = ROOT+"data/data_process/Model_A/"
    tmp_root = ROOT+"data/data_process/Model_A/tmp/"
    label_root = ROOT+"data/data_process/labels_single/"

    mutex=threading.Lock()
    paths=[]
    parallel_dict={"loop_cnts":0,"last_timestamp":None}

    for path_name in walk(raw_root):
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
                    print("Loops:{}/{}, {}%, speed:{}, ETA:{}s".format(parallel_dict["loop_cnts"],len(paths),percentage,speed,ETA))
                parallel_dict["last_timestamp"]=time.time()
            mutex.release()
    
            path_RPS = paths[idx]
            colocation, name, n_thread, RPS = path_RPS.split('/')[-4:]

            tmp_dir = tmp_root+"{}/{}/{}/".format(colocation,name,n_thread)
            tmp_path = tmp_dir+"{}.csv".format(RPS)

            if os.path.exists(tmp_path):
                continue
            labels = get_labels(path_RPS, label_root)
            if labels is None:
                continue
            else:
                OAA_df_path=path_RPS + "/{}+{}+{}+{}+{}".format(name, n_thread, RPS, labels["OAA_Core"], cache_2_way(labels["OAA_Cache"], MB_PER_WAY))
                if not os.path.exists(OAA_df_path):
                    continue
                OAA_df = pd.read_csv(OAA_df_path)
                OAA_bandwidth = OAA_df["MBL"].mean()
                labels["OAA_Bandwidth"] = OAA_bandwidth

            RPS_df=pd.DataFrame()
            for path_case in walk(path_RPS):
                try:
                    df = pd.read_csv(path_case)
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

def process_A_shadow(n_threads=N_CORES):
    raw_root = ROOT+"data/data_collection/multiple/"
    data_root = ROOT+"data/data_process/Model_A_shadow/"
    tmp_root = ROOT+"data/data_process/Model_A_shadow/tmp/"
    label_root = ROOT+"data/data_process/labels_multiple/" 

    mutex=threading.Lock()
    paths=[]
    parallel_dict={"loop_cnts":0,"last_timestamp":None}

    for path_name in walk(raw_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                paths.append(path_RPS)

    def t_process_A_shadow(paths,idx_start,idx_end,parallel_dict):
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
            labels = get_labels(path_RPS, label_root)
            if labels is None:
                continue
            else:
                path_read = path_RPS + "/{}+{}+{}+{}+{}".format(name, n_thread, RPS, labels["OAA_Core"], cache_2_way(labels["OAA_Cache"], MB_PER_WAY))
                if not os.path.exists(path_read):
                    continue
                OAA_df = pd.read_csv(path_read)
                OAA_bandwidth = OAA_df["MBL"].mean()
                labels["OAA_Bandwidth"] = OAA_bandwidth

            RPS_df=pd.DataFrame()
            for path_case in walk(path_RPS):
                try:
                    df = pd.read_csv(path_case)
                    df = df[A_SHADOW_FEATURES]
                except pd.errors.EmptyDataError as e:
                    print(path_case, ' is empty')
                    os.system('rm -f {}'.format(path_case))
                RPS_df = pd.concat([RPS_df,df], ignore_index=True, sort=False)
            
            for key in labels:
                RPS_df[key] = labels[key]
            RPS_df=RPS_df[A_SHADOW_FEATURES+A_LABELS]
            os.system("mkdir -p {}".format(tmp_dir))
            RPS_df.to_csv(tmp_path, index=None)
    
    threads=[]
    blk_length=len(paths)//n_threads+1
    for t_idx in range(n_threads):
        idx_start=t_idx*blk_length
        idx_end=(t_idx+1)*blk_length
        thread=threading.Thread(target=t_process_A_shadow,args=(paths,idx_start,idx_end,parallel_dict))
        threads.append(thread)
    for t in threads:
        t.start() 

def process_B():
    raw_root = ROOT+"data/data_collection/multiple/"
    data_root = ROOT+"data/data_process/Model_B/"
    tmp_root = ROOT+"data/data_process/Model_B/tmp/" 
    label_root = ROOT+"data/data_process/labels_multiple/"
    names = {}
    for path_name in walk(raw_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                name, n_thread, RPS = path_RPS.split('/')[-3:]
                if name in names:
                    names[name].append(path_RPS)
                else:
                    names[name] = [path_RPS]

    for name in names.keys():
        whole_queue = multiprocessing.Manager().Queue(len(names[name]))
        df_lock = multiprocessing.Manager().Lock()
        for item in names[name]:
            whole_queue.put(item)
        pool = multiprocessing.Pool(os.cpu_count())
        args = [[whole_queue]] * min(os.cpu_count(), len(names[name]))
        pool.map_async(process_B_process, args)
        pool.close()
        pool.join()

def process_B_process(args):
    raw_root = ROOT+"data/data_collection/multiple/"
    data_root = ROOT+"data/data_process/Model_B/"
    tmp_root = ROOT+"data/data_process/Model_B/tmp/" 
    label_root = ROOT+"data/data_process/labels_multiple/"
    whole_queue = args[0]
    while whole_queue.qsize() != 0:
        try:
            path_RPS = whole_queue.get_nowait()
        except whole_queue.Empty:
            break
        print(path_RPS)
        colocation, name, n_thread, RPS = path_RPS.split('/')[-4:]
        tmp_dir = tmp_root+"{}/{}/{}/".format(colocation,name,n_thread)
        tmp_path = tmp_dir+"{}.csv".format(RPS)

        if os.path.exists(tmp_path):
            continue
        RPS_df=pd.DataFrame()
        labels = get_labels(path_RPS, label_root)
        if labels is None:
            continue
        OAA_core_idx = labels["OAA_Core"] - 1
        OAA_way_idx = cache_2_way(labels["OAA_Cache"], MB_PER_WAY) - 1
        lat_df = pd.DataFrame(columns=WAY_INDEX, index=CORE_INDEX)
        dfs = {}
        for core_idx in range(N_CORES):
            for way_idx in range(N_WAYS):
                path_case = path_RPS + "/" + "{}+{}+{}+{}+{}".format(name, n_thread, RPS, core_idx, way_idx)
                if not os.path.exists(path_case):
                    continue
                df = pd.read_csv(path_case)
                avg_latency = df['Latency'].mean()
                lat_df.loc[core_idx, way_idx] = avg_latency
                dfs[(core_idx, way_idx)] = df

        l = 10
        core_start = OAA_core_idx - l if OAA_core_idx - l >= 0 else 0
        core_end = OAA_core_idx + l if OAA_core_idx + l <= N_CORES - 1 else N_CORES - 1
        way_start = OAA_way_idx - l if OAA_way_idx - l >= 0 else 0
        way_end = OAA_way_idx + l if OAA_way_idx + l <= N_WAYS - 1 else N_WAYS - 1
        for core_idx_allocated in range(core_end, core_start - 1, -1):
            for way_idx_allocated in range(way_end, way_start - 1, -1):
                latency_allocated = lat_df.loc[core_idx_allocated, way_idx_allocated]
                
                QoS = QOS_TARGET[name]/latency_allocated
                bounds = [(QOS_TARGET[name]/(QoS - slowdown), slowdown) for slowdown in [0.05,0.1,0.15,0.2] if QoS-slowdown>0.01]
                for _, tpl in enumerate(bounds):
                    if (core_idx_allocated, way_idx_allocated) not in dfs:
                        continue
                    bound=tpl[0]
                    slowdown=tpl[1]
                    df = deepcopy(dfs[(core_idx_allocated, way_idx_allocated)])
                    vis = []
                    target = {key: 0 for key in B_LABELS}
                    for core_idx in range(core_idx_allocated, -1, -1):  # Core dominated
                        for way_idx in range(way_idx_allocated, max(-1,way_idx_allocated-4),-1):
                            if lat_df.loc[core_idx, way_idx_allocated] <= bound and (core_idx, way_idx_allocated ) not in vis:
                                target["Core_Dominated_Spared_Core"] = core_idx_allocated - core_idx
                                target["Core_Dominated_Spared_Cache"] = way_2_cache(way_idx_allocated - way_idx, MB_PER_WAY)
                                vis.append((core_idx, way_idx_allocated))
                    for way_idx in range(way_idx_allocated, -1, -1):  # Way dominated
                        for core_idx in range(core_idx_allocated,max(-1,core_idx_allocated-4),-1):
                            if lat_df.loc[core_idx_allocated, way_idx] <= bound and (core_idx_allocated, way_idx) not in vis:
                                target["Cache_Dominated_Spared_Core"] = core_idx_allocated - core_idx
                                target["Cache_Dominated_Spared_Cache"] = way_2_cache(way_idx_allocated - way_idx, MB_PER_WAY)
                                vis.append((core_idx_allocated, way_idx))
                    for core_idx in range(core_idx_allocated, -1, -1):  # Default
                        for way_idx in range(way_idx_allocated, -1, -1):
                            if lat_df.loc[core_idx, way_idx] <= bound and (core_idx, way_idx) not in vis:
                                target["Default_Spared_Core"] = core_idx_allocated - core_idx
                                target["Default_Spared_Cache"] = way_2_cache(way_idx_allocated - way_idx, MB_PER_WAY)
                                vis.append((core_idx, way_idx))
                    df["QoS_Reduction"] = slowdown
                    for key in target:
                        df[key] = target[key]
                    RPS_df = pd.concat([RPS_df,df], ignore_index=True, sort=False)
        print(RPS_df)
        RPS_df=RPS_df[B_FEATURES+B_LABELS]
        os.system("mkdir -p {}".format(tmp_dir))
        RPS_df.to_csv(tmp_path, index=None)
        print(str(os.getpid()) + " finished " + path_RPS)

def process_B_shadow():
    raw_root = ROOT+"data/data_collection/multiple/"
    data_root = ROOT+"data/data_process/Model_B_shadow/"
    tmp_root = ROOT+"data/data_process/Model_B_shadow/tmp/" 
    label_root = ROOT+"data/data_process/labels_multiple/"
    names = {}
    for path_name in walk(raw_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                name, n_thread, RPS = path_RPS.split('/')[-3:]
                if name in names:
                    names[name].append(path_RPS)
                else:
                    names[name] = [path_RPS]

    for name in names.keys():
        whole_queue = multiprocessing.Manager().Queue(len(names[name]))
        df_lock = multiprocessing.Manager().Lock()
        for item in names[name]:
            whole_queue.put(item)
        pool = multiprocessing.Pool(os.cpu_count())
        args = [[whole_queue]] * os.cpu_count()
        pool.map_async(process_B_shadow_process, args)
        pool.close()
        pool.join()
    
def process_B_shadow_process(args):
    raw_root = ROOT+"data/data_collection/multiple/"
    data_root = ROOT+"data/data_process/Model_B_shadow/"
    tmp_root = ROOT+"data/data_process/Model_B_shadow/tmp/" 
    label_root = ROOT+"data/data_process/labels_multiple/"
    whole_queue = args[0]
    while whole_queue.qsize() != 0:
        try:
            path_RPS = whole_queue.get_nowait()
        except whole_queue.Empty:
            break
        print(path_RPS)
        colocation, name, n_thread, RPS = path_RPS.split('/')[-4:]
        tmp_dir = tmp_root+"{}/{}/{}/".format(colocation,name,n_thread)
        tmp_path = tmp_dir+"{}.csv".format(RPS)

        if os.path.exists(tmp_path):
            continue

        RPS_df=pd.DataFrame()
        labels = get_labels(path_RPS, label_root)
        if labels is None:
            continue
        OAA_core_idx = labels["OAA_Core"] - 1
        OAA_way_idx = cache_2_way(labels["OAA_Cache"], MB_PER_WAY) - 1
        lat_df = pd.DataFrame(columns=WAY_INDEX, index=CORE_INDEX)
        dfs = {}
        for path_case in walk(path_RPS):
            core_idx, way_idx = path_case.split('/')[-1].split('+')[3:]
            core_idx, way_idx = int(core_idx), int(way_idx)
            df = pd.read_csv(path_case)
            avg_latency = df['Latency'].mean()
            lat_df.loc[core_idx, way_idx] = avg_latency
            dfs[(core_idx, way_idx)] = df
        l = 4
        core_start = OAA_core_idx - l if OAA_core_idx - l >= 0 else 0
        core_end = OAA_core_idx + l if OAA_core_idx + l <= N_CORES - 1 else N_CORES - 1
        way_start = OAA_way_idx - l if OAA_way_idx - l >= 0 else 0
        way_end = OAA_way_idx + l if OAA_way_idx + l <= N_WAYS - 1 else N_WAYS - 1
        for core_idx_allocated in range(core_end, core_start - 1, -1):
            for way_idx_allocated in range(way_end, way_start - 1, -1):
                if (core_idx_allocated, way_idx_allocated) not in dfs:
                    continue
                for core_idx_target in range(core_idx_allocated, core_start - 1, -1):
                    for way_idx_target in range(way_idx_allocated, way_start - 1, -1):
                        df = dfs[(core_idx_allocated, way_idx_allocated)].copy()
                        if core_idx_target == core_idx_allocated and way_idx_target == way_idx_allocated:
                            continue
                        latency_allocated = lat_df.loc[core_idx_allocated, way_idx_allocated]
                        latency_target = lat_df.loc[core_idx_target, way_idx_target]
                        qos_reduction = (QOS_TARGET[name]/latency_target)-(QOS_TARGET[name]/latency_allocated)
                        df['Target_Cache'] = way_2_cache(way_idx_target + 1, MB_PER_WAY)
                        df['Target_Core'] = core_idx_target + 1
                        df['QoS_Reduction'] = qos_reduction
                        RPS_df = pd.concat([RPS_df,df], ignore_index=True, sort=False)
        RPS_df=RPS_df[B_SHADOW_FEATURES+B_SHADOW_LABELS]
        os.system("mkdir -p {}".format(tmp_dir))
        RPS_df.to_csv(tmp_path, index=None)
        print(str(os.getpid()) + " finished " + path_RPS)

def model_C_reward(Latency, Latency_, Action_ID):
    delta_latency=Latency_-Latency
    Action_ID=int(Action_ID)
    action={"cores":ACTION_SPACE[Action_ID][0],"ways":ACTION_SPACE[Action_ID][1]}
    if delta_latency>0:
        reward = -np.log(1+delta_latency) - (action["cores"]+action["ways"])
    elif delta_latency<0:
        reward = np.log(1-delta_latency) - (action["cores"]+action["ways"])
    else:
        reward = - (action["cores"]+action["ways"])
    return round(reward,5)

def process_C_process(args):
    whole_queue = args[0]
    whole_bench_df = args[1]
    df_lock = args[2]
    while whole_queue.qsize() != 0:
        try:
            path_RPS = whole_queue.get_nowait()
        except whole_queue.Empty:
            break
        print(path_RPS)
        colocation, name, n_thread, RPS = path_RPS.split('/')[-4:]
        tmp_root = ROOT+"data/data_process/Model_C/tmp/"
        tmp_dir = tmp_root+"{}/{}/{}/".format(colocation,name,n_thread)
        tmp_path = tmp_dir+"{}.csv".format(RPS)
        if os.path.exists(tmp_path):
            continue
        RPS_df=pd.DataFrame()
        dfs = {}
        for path_case in walk(path_RPS):
            core_idx, way_idx = path_case.split('/')[-1].split('+')[3:]
            core_idx, way_idx = int(core_idx), int(way_idx)
            df = pd.read_csv(path_case)
            df = df.sample(n=5)
            df = df.round({'Frequency': 2, 'Latency': 2})
            dfs[(core_idx, way_idx)] = df
        for core_idx_allocated in range(N_CORES):
            for way_idx_allocated in range(N_WAYS):
                allocated_df = pd.DataFrame()
                if (core_idx_allocated, way_idx_allocated) not in dfs or dfs[(core_idx_allocated, way_idx_allocated)].empty:
                    continue
                l = 3
                core_start = core_idx_allocated - l if core_idx_allocated - l >= 0 else 0
                core_end = core_idx_allocated + l if core_idx_allocated + l <= N_CORES - 1 else N_CORES - 1
                way_start = way_idx_allocated - l if way_idx_allocated - l >= 0 else 0
                way_end = way_idx_allocated + l if way_idx_allocated + l <= N_WAYS - 1 else N_WAYS - 1
                for core_idx_target in range(core_start, core_end + 1):
                    for way_idx_target in range(way_start, way_end + 1):
                        if (core_idx_target, way_idx_target) not in dfs or dfs[(core_idx_target, way_idx_target)].empty:
                            continue
                        action = (core_idx_target - core_idx_allocated, way_idx_target - way_idx_allocated)
                        action_id = ACTION_ID[action]
                        df_allocated = dfs[(core_idx_allocated, way_idx_allocated)].copy()[C_FEATURES["s"]]
                        df_target = dfs[(core_idx_target, way_idx_target)].copy()[C_FEATURES["s"]]
                        df_target.columns = C_FEATURES["s_"]
                        df_allocated[C_FEATURES["a"]] = action_id
                        df_target[C_FEATURES["a"]] = action_id
                        df = pd.merge(df_allocated, df_target, on=C_FEATURES["a"])
                        allocated_df = pd.concat([allocated_df,df], ignore_index=True, sort=False)
                allocated_df[C_FEATURES["r"][0]] = allocated_df.apply(lambda x: model_C_reward(x.Latency, x.Latency_, x.Action_ID), axis=1)
                allocated_df=allocated_df[C_FEATURES["s"]+C_FEATURES["a"]+C_FEATURES["r"]+C_FEATURES["s_"]]
                RPS_df = pd.concat([RPS_df,allocated_df], ignore_index=True, sort=False)
        os.system("mkdir -p {}".format(tmp_dir))
        RPS_df.to_csv(tmp_path, index=None)
        print(str(os.getpid()) + " finished " + path_RPS)


def process_C():
    names = {}
    bench_df = pd.DataFrame()
    for root in ["single","multiple"]:
        raw_root = ROOT+"data/data_collection/{}/".format(root)
        data_root = ROOT+"data/data_process/Model_C/{}/".format(root)
        for path_name in walk(raw_root):
            for path_thread in walk(path_name):
                for path_RPS in walk(path_thread):
                    name, n_thread, RPS = path_RPS.split('/')[-3:]
                    if name in names:
                        names[name].append(path_RPS)
                    else:
                        names[name] = [path_RPS]
    for name in names.keys():
        whole_queue = multiprocessing.Manager().Queue(len(names[name]))
        df_lock = multiprocessing.Manager().Lock()
        for item in names[name]:
            whole_queue.put(item)
        pool = multiprocessing.Pool(min(os.cpu_count(), len(names[name])))
        args = [[whole_queue, bench_df, df_lock]] * min(os.cpu_count(), len(names[name]))
        pool.map_async(process_C_process, args)
        pool.close()
        pool.join()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ["Model_A", "Model_A_shadow", "Model_B", "Model_B_shadow", "Model_C"]
    func_map = {"Model_A": process_A, "Model_A_shadow": process_A_shadow, "Model_B": process_B, "Model_B_shadow": process_B_shadow,"Model_C":process_C}
    for dataset in datasets:
        func_map[dataset]()
