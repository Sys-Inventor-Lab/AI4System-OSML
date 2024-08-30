import pandas as pd
import numpy as np
import sys
import pickle
from tqdm import tqdm
from copy import deepcopy
from annotation import annotation,get_labels

sys.path.append("../../")
from utils import *
from configs import *

def merge_offline(dataset, data_source):
    all_lines=[]
    data_root = ROOT+"data/data_process/{}/tmp/{}/".format(dataset, data_source)
    result_path = ROOT+"data/data_process/{}/{}.csv".format(dataset,dataset)
    for path_name in tqdm(walk(data_root)):
        print(path_name)
        for path_thread in tqdm(walk(path_name)):
            for path_RPS in walk(path_thread):
                if os.path.exists(path_RPS):
                    with open(path_RPS,"r") as f:
                        f_lines=f.readlines()
                        if len(all_lines)==0:
                            all_lines.append(f_lines)
                        else:
                            all_lines.append(f_lines[1:])
    result_lines=[]
    for lines in all_lines:
        result_lines+=lines
    with open(result_path,"w") as f:
        f.writelines(result_lines)

def merge_online(dataset, data_source):
    for source in data_source:
        data_root = ROOT+"data/data_process/{}/tmp/{}/".format(dataset, source)
        for path_name in tqdm(walk(data_root)):
            all_lines=[]
            name=path_name.split("/")[-1]
            result_path = ROOT+"data/data_process/{}/{}_{}_{}.csv".format(dataset,dataset,source,name)
            for path_thread in tqdm(walk(path_name)):
                for path_RPS in walk(path_thread):
                    if os.path.exists(path_RPS):
                        with open(path_RPS,"r") as f:
                            f_lines=f.readlines()
                            if len(all_lines)==0:
                                all_lines.append(f_lines)
                            else:
                                all_lines.append(f_lines[1:])
            result_lines=[]
            for lines in all_lines:
                result_lines+=lines
            with open(result_path,"w") as f:
                f.writelines(result_lines)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets=sys.argv[1:]
    else:
        datasets=["Model_A","Model_A_shadow","Model_B","Model_B_shadow","Model_C"]
    func_map={"Model_A":merge_offline,"Model_A_shadow":merge_offline,"Model_B":merge_offline,"Model_B_shadow":merge_offline,"Model_C":merge_online}
    data_source={"Model_A":"single","Model_A_shadow":"multiple","Model_B":"multiple","Model_B_shadow":"multiple","Model_C":["single","multiple"]}
    for dataset in datasets:
        func_map[dataset](dataset,data_source[dataset])
