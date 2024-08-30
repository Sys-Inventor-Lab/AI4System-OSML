import sys
sys.path.append("../../")
from utils import *
from configs import *
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def main():
    ALL_FEATURES=["CPU_Utilization", "Frequency", "IPC", "Misses", "MBL", "Virt_Memory", "Res_Memory", "Allocated_Cache", "Allocated_Core", "MBL_N", "Allocated_Cache_N", "Allocated_Core_N", "Latency", "Target_Cache", "Target_Core", "QoS_Reduction"]
    RAW_ROOT = [ROOT+"/data/data_collection/single/", ROOT+"/data/data_collection/multiple/"]
    max_min = {"max":defaultdict(float), "min":defaultdict(float)}

    dataset_paths=[ ROOT+"/data/data_process/Model_A/Model_A.csv",
                    ROOT+"/data/data_process/Model_A_shadow/Model_A_shadow.csv",
                    ROOT+"/data/data_process/Model_B/Model_B.csv",
                    ROOT+"/data/data_process/Model_B_shadow/Model_B_shadow.csv"]
    for path in walk(ROOT+"/data/data_process/Model_C/"):
        if path.endswith(".csv"):
            dataset_paths.append(path)

    print(dataset_paths)
    for path in tqdm(dataset_paths):
        df=pd.read_csv(path)
        df_features=df.columns.to_list()
        df_max=df.max().to_numpy()
        df_min=df.min().to_numpy()
        for _,feature in enumerate(df_features):
            if feature in ALL_FEATURES:
                max_min["max"][feature]=max(max_min["max"][feature],df_max[_])
                max_min["min"][feature]=min(max_min["min"][feature],df_min[_])

    max_min["max"]=dict(max_min["max"])
    max_min["min"]=dict(max_min["min"])
    with open(ROOT+"/data/data_process/max_min.txt","w") as f:
        f.write(str(max_min))

if __name__=="__main__":
    main()
