import pandas as pd
import numpy as np
import sys
import os
import time

sys.path.append("../")
from program_mgr import program_mgr
from configs import *

ROOT_DIR = "data_collection/single/"
NAMES = ["mongodb", "img-dnn", "xapian", "sphinx", "specjbb", "masstree", "login", "moses", "memcached"]
MAX_THREADS = 36


def collect(mgr, name, interval, n_records):
    df = pd.DataFrame(columns=COLLECT_FEATURES)
    if name=="nginx":
        print("sleep")
        time.sleep(3)
    for i in range(n_records):
        time.sleep(interval)
        arr = mgr.get_features(name, COLLECT_FEATURES)
        if None in arr:
            continue
        arr = [float(item) for item in arr]
        print(arr)
        df.loc[df.shape[0]] = arr
    print(df)
    return df


def main():
    for name in NAMES:
        for thread in reversed(range(1, MAX_THREADS + 1)):
            for RPS in RPS_COLLECTED[name]:
                data_dir = ROOT_DIR + "{}/{}/{}/".format(name, thread, RPS)
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                for core_idx in reversed(range(N_CORES)):
                    for way_idx in reversed(range(N_WAYS)):
                        file_name = data_dir + "+".join([str(i) for i in [name, thread, RPS, core_idx, way_idx]])
                        if os.path.exists(file_name):
                            continue
                        try:
                            print(file_name)
                            mgr = program_mgr(enable_models=False)
                            mgr.add_RPS(name, RPS, thread)
                            mgr.launch(name)
                            mgr.allocate(name, {"cores": core_idx + 1, "ways": way_idx + 1})
                            df = collect(mgr, name, 1, 10)
                            mgr.end_all()
                            df.to_csv(file_name,index=False)
                        except Exception as e:
                            with open("error_log.txt", "a") as f:
                                f.write(str([name, thread, RPS, core_idx, way_idx]) + '\n')
                                f.write(str(e))
                            raise e
                        os.system(ROOT + "/reset.sh {}".format(ROOT + "/"))
                        time.sleep(1)
                        
if __name__ == "__main__":
    main()
