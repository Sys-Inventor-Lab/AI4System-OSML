import pandas as pd
import sys
from tqdm import tqdm
import numpy as np

sys.path.append("../")
from utils import *


def clean_data_offline(dataset):
    path="data_process/{}/{}.csv".format(dataset,dataset)
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df = df.dropna()
    df.to_csv(path, index=None)

def clean_data_online(dataset):
    path="data_process/{}/".format(dataset)
    for path_case in tqdm(walk(path)):
        print(path_case)
        df=pd.read_csv(path_case)
        df=df.dropna()
        df.to_csv(path_case, index=None)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ["Model_A","Model_A_shadow","Model_B","Model_B_shadow","Model_C"]
    func_map={"Model_A":clean_data_offline,"Model_A_shadow":clean_data_offline,"Model_B":clean_data_offline,"Model_B_shadow":clean_data_offline,"Model_C":clean_data_online}

    for dataset in datasets:
        func_map[dataset](dataset)
