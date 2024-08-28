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

def merge_A(dataset):
    all_df = pd.DataFrame()
    data_root = ROOT+"/data/data_process/{}/tmp/multiple/".format(dataset)
    for path_name in tqdm(walk(data_root)):
        for path_thread in tqdm(walk(path_name)):
            for path_RPS in walk(path_thread):
                if os.path.exists(path_RPS):
                    try:
                        df=pd.read_csv(path_RPS)
                    except pd.errors.EmptyDataError:
                        continue
                    all_df = pd.concat([all_df,df],ignore_index=True,sort=False)
    all_df.to_csv(ROOT+"/data/data_process/{}/{}.csv".format(dataset,dataset),index=None)

def merge_B(dataset):
    all_df = pd.DataFrame()
    data_root = ROOT+"/data/data_process/{}/tmp/multiple/".format(dataset)
    for path_name in tqdm(walk(data_root)):
        for path_thread in tqdm(walk(path_name)):
            for path_RPS in walk(path_thread):
                if os.path.exists(path_RPS):
                    try:
                        df=pd.read_csv(path_RPS)
                    except pd.errors.EmptyDataError:
                        continue
                    all_df = pd.concat([all_df,df],ignore_index=True,sort=False)
    all_df.to_csv(ROOT+"/data/data_process/{}/{}.csv".format(dataset,dataset),index=None)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets=sys.argv[1:]
    else:
        datasets=["Model_A","Model_B"]
    func_map={"Model_A":merge_A,"Model_B":merge_B}
    for dataset in datasets:
        func_map[dataset](dataset)
