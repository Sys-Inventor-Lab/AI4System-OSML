import pandas as pd
import sys
from tqdm import tqdm
import numpy as np

sys.path.append("../../")
from utils import *
from configs import *


def clean_data_process(path):
    if not os.path.exists(path + ".csv"):
        return
    df = pd.read_csv(path + ".csv")
    df = df.dropna()
    df.to_csv(path + ".csv", index=None)


def clean_data_collection():
    root=ROOT+"/data/data_collection/single/memcached/"
    for path_thread in walk(root):
        for path_RPS in walk(path_thread):
            name, n_thread, RPS = path_RPS.split('/')[-3:]
            for path_case in walk(path_RPS):
                df = pd.read_csv(path_case, index_col=0)
                if (np.isnan(df["Latency"].mean())):
                    print(df)
                    print(path_case)
                    os.system("rm "+path_case)


def clean_label():
    root=ROOT+"/data/data_process/labels_single/"
    for path_name in walk(root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                if os.path.exists(path_RPS+"/labels.txt"):
                    with open(path_RPS+"/labels.txt","r") as f:
                        line=f.readline()
                        print(line)

def clean_index():
    raw_root = ROOT+"/data/data_collection/single/"

    for path_name in tqdm(walk(raw_root)):
        for path_thread in tqdm(walk(path_name)):
            for path_RPS in walk(path_thread):
                name, n_thread, RPS = path_RPS.split('/')[-3:]
                for path_case in walk(path_RPS):
                    try:
                        head_line=subprocess.check_output("head -n 1 {}".format(path_case), shell=True).decode()
                        if head_line[0]==",":
                            print("has index")
                            df = pd.read_csv(path_case, index_col=0)
                            df.to_csv(path_case,index=False)
                    except pd.errors.EmptyDataError as e:
                        print(path_case, ' is empty')
                        os.system('rm -f {}'.format(path_case))


if __name__ == '__main__':
    '''
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ["Model_A","Model_A_shadow","Model_B","Model_B_shadow","Model_C"]
    for dataset in datasets:
        path = ROOT+"data/data_process/{}/{}".format(dataset, dataset)
        clean_data_process(path)
        clean_data_collection(path)
    '''
    #clean_label()
    #clean_index()
    clean_data_collection()
