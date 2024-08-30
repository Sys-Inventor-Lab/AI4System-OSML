import pandas as pd
import sys
import subprocess
sys.path.append("../../")
from utils import *

def count_samples_A():
    n_samples=0
    tmp_root = ROOT+"data/data_process/Model_A/tmp/single/"
    for path_name in walk(tmp_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                n_samples += int(subprocess.check_output("wc -l {}".format(path_RPS), shell=True).decode().split()[0]) - 1 
    print("Model A:")
    print("Samples:", n_samples)

def count_samples_A_shadow():
    n_samples=0
    tmp_root = ROOT+"data/data_process/Model_A_shadow/tmp/multiple/"
    for path_name in walk(tmp_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                n_samples += int(subprocess.check_output("wc -l {}".format(path_RPS), shell=True).decode().split()[0]) - 1 
    print("Model A shadow:")
    print("Samples:", n_samples)
    

def count_samples_B():
    n_samples=0
    tmp_root = ROOT+"data/data_process/Model_B/tmp/multiple/"
    for path_name in walk(tmp_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                n_samples += int(subprocess.check_output("wc -l {}".format(path_RPS), shell=True).decode().split()[0]) - 1 
    print("Model B:")
    print("Samples:", n_samples)

def count_samples_B_shadow():
    n_samples=0
    tmp_root = ROOT+"data/data_process/Model_B_shadow/tmp/multiple/"
    for path_name in walk(tmp_root):
        for path_thread in walk(path_name):
            for path_RPS in walk(path_thread):
                n_samples += int(subprocess.check_output("wc -l {}".format(path_RPS), shell=True).decode().split()[0]) - 1 
    print("Model B shadow:")
    print("Samples:", n_samples)

def count_samples_C():
    n_samples=0
    root = ROOT+"data/data_process/Model_C/"
    for path_case in walk(root):
        n_samples += int(subprocess.check_output("wc -l {}".format(path_case), shell=True).decode().split()[0]) - 1 

    print("Model C:")
    print("Samples:", n_samples)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ["Model_A", "Model_A_shadow", "Model_B", "Model_B_shadow", "Model_C"]
    func_map = {"Model_A": count_samples_A, "Model_A_shadow": count_samples_A_shadow, "Model_B": count_samples_B, "Model_B_shadow": count_samples_B_shadow,"Model_C":count_samples_C}
    for dataset in datasets:
        func_map[dataset]()
