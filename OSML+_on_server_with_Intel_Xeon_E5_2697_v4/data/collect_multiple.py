import pandas as pd
import numpy as np
import time
import sys
import os
import random

sys.path.append("../")
from program_mgr import program_mgr
from configs import *

random.seed(1)

NAMES = ["img-dnn", "xapian", "sphinx", "specjbb", "masstree", "moses", "mongodb"]
COLOCATE_NAMES = ["img-dnn", "xapian", "sphinx", "specjbb", "masstree"]
DATA_ROOT = "data_collection/multiple/"
CONFIG_ROOT = "data_collection/configs_multiple/"
THREADS = [N_CORES]


def select_colocated_applications(exclude):
    config = {}
    apps_remain = COLOCATE_NAMES.copy()
    if exclude in apps_remain:
        apps_remain.remove(exclude)
    noise_num = random.randint(0, min(4, len(apps_remain)))
    noise_apps = random.sample(apps_remain, noise_num)
    # Generate RPS of each application
    for app in noise_apps:
        config[app] = int(MAX_LOAD[app] * random.random())
    return config

def collect(mgr, name, interval, n_records):
    df = pd.DataFrame(columns=COLLECT_MUL_FEATURES)
    for i in range(n_records):
        time_remaining = interval - time.time() % interval
        time.sleep(time_remaining)
        arr = mgr.get_colocated_features(name)
        print(arr)
        df.loc[df.shape[0]] = arr
    return df


def main():
    # For all applications
    for name in NAMES:
        # For all threads to be collected
        for thread in THREADS:
            # For all RPS to be collected
            for pct in [20,40,60,80,100]:
                RPS = int(MAX_LOAD[name]*pct/100)
                # Read or randomly select applications to be co-located
                data_dir = DATA_ROOT + "{}/{}/{}/".format(name, thread, RPS)
                config_dir = CONFIG_ROOT + "{}/{}/{}/".format(name, thread, RPS)
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                if os.path.exists(config_dir + "config.txt"):
                    # Read configuration from saved file
                    with open(config_dir + "config.txt", "r") as f:
                        config = eval(f.readline())
                else:
                    os.makedirs(config_dir)
                    # Randomly select applications to be co-located
                    config = select_colocated_applications(exclude=name)
                    with open(config_dir + "config.txt", "w") as f:
                        f.write(str(config))

                # Collect all possible resource allocation policies
                for core_idx in range(N_CORES):
                    for way_idx in range(N_WAYS):
                        file_name = data_dir + "+".join([str(i) for i in [name, thread, RPS, core_idx, way_idx]])
                        if os.path.exists(file_name):
                            continue
                        try:
                            print(file_name)
                            print(config)
                            colocate_enabled = not (core_idx == N_CORES - 1 or way_idx == N_WAYS - 1)
                            mgr = program_mgr()
                            mgr.add_app(name, "RPS", RPS, thread)
                            if colocate_enabled:
                                for colocate_app in config:
                                    mgr.add_app(colocate_app, "RPS", config[colocate_app])
                            mgr.launch_all()
                            if colocate_enabled:
                                mgr.allocate(name, {"cores": core_idx + 1, "ways": way_idx + 1}, propagate=False)
                                mgr.allocate_sharing(names=list(config.keys()),
                                                     allocation={"cores": N_CORES - core_idx - 1,
                                                                 "ways": N_WAYS - way_idx - 1}, propagate=True)
                            else:
                                mgr.allocate(name, {"cores": core_idx + 1, "ways": way_idx + 1}, propagate=True)
                            mgr.report_allocation()
                            df = collect(mgr, name, 0.1, 60)
                            mgr.end_all()
                            df.to_csv(file_name,index=False)
                        except Exception as e:
                            with open("error_log.txt", "a") as f:
                                f.write(str([name, thread, RPS, core_idx, way_idx]) + '\n')
                                f.write(str(e))
                        os.system(ROOT + "/reset.sh {}".format(ROOT + "/"))
                        time.sleep(1)


if __name__ == "__main__":
    main()
