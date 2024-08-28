import pandas as pd
import sys
sys.path.append("../../")
from utils import *
from configs import *

def count_samples_A():
    n_samples=0
    n_cases=0
    tmp_root = ROOT+"/data/data_process/Model_A/tmp/single/"
    for path_name in walk(tmp_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                df=pd.read_csv(path_RPS)
                n_samples+=df.shape[0]
                print(df.value_counts(["Allocated_Cache", "Allocated_Core"]))
                n_cases+=df.value_counts(["Allocated_Cache","Allocated_Core"]).shape[0]
    print("Model A:")
    print("Samples:", n_samples)
    print("Cases:", n_cases)

def count_samples_A_shadow():
    n_samples=0
    n_cases=0
    tmp_root = ROOT+"/data/data_process/Model_A_shadow/tmp/multiple/"
    for path_name in walk(tmp_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                df=pd.read_csv(path_RPS)
                n_samples+=df.shape[0]
                n_cases+=df.value_counts(["Allocated_Cache","Allocated_Core"]).shape[0]
    print("Model A shadow:")
    print("Samples:", n_samples)
    print("Cases:", n_cases)
    

def count_samples_B():
    n_samples=0
    n_cases=0
    tmp_root = ROOT+"/data/data_process/Model_B/tmp/multiple/"
    for path_name in walk(tmp_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                df=pd.read_csv(path_RPS)
                n_samples+=df.shape[0]
                n_cases+=df.value_counts(["Allocated_Cache","Allocated_Core","QoS_Reduction"]).shape[0]
    print("Model B:")
    print("Samples:", n_samples)
    print("Cases:", n_cases)

def count_samples_B_shadow():
    n_samples=0
    n_cases=0
    tmp_root = ROOT+"/data/data_process/Model_B_shadow/tmp/multiple/"
    for path_name in walk(tmp_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                df=pd.read_csv(path_RPS)
                n_samples+=df.shape[0]
                n_cases+=df.value_counts(["Allocated_Cache","Allocated_Core","Target_Cache","Target_Core"]).shape[0]
    print("Model B shadow:")
    print("Samples:", n_samples)
    print("Cases:", n_cases)

def count_samples_C():
    n_samples=0
    n_cases=0
    for root in ["single","multiple"]:
        tmp_root = ROOT+"/data/data_process/Model_C/tmp/{}/".format(root)
        for path_name in walk(tmp_root):
            for path_thread in walk(path_name):
                for path_RPS in walk(path_thread):
                    try:
                        df=pd.read_csv(path_RPS)
                    except pd.errors.EmptyDataError as e:
                        continue
                    n_samples+=df.shape[0]
                    n_cases+=df.value_counts(["Allocated_Cache","Allocated_Core","Action_ID"]).shape[0]
    print("Model C:")
    print("Samples:", n_samples)
    print("Cases:", n_cases)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ["Model_A", "Model_A_shadow", "Model_B", "Model_B_shadow", "Model_C"]
    func_map = {"Model_A": count_samples_A, "Model_A_shadow": count_samples_A_shadow, "Model_B": count_samples_B, "Model_B_shadow": count_samples_B_shadow,"Model_C":count_samples_C}
    for dataset in datasets:
        func_map[dataset]()
