import pandas as pd
import numpy as np
import sys
import os
import time
import subprocess
sys.path.append("../../")
from program_mgr import program_mgr
from configs import *
from utils import *

ROOT_DIR = ROOT+"/data/data_collection/single/"
NAMES = ["mongodb", "img-dnn", "xapian", "sphinx", "specjbb", "masstree", "login", "moses", "memcached"]
MAX_THREADS = 36


def main():
    for path_1 in walk(ROOT_DIR):
        for path_2 in walk(path_1):
            for path_3 in walk(path_2):
                print(path_3)
                for path_4 in walk(path_3):
                    n=int(subprocess.check_output("wc -l {}".format(path_4),shell=True).decode().split()[0])
                    if n==0:
                        print(path_4)
if __name__ == "__main__":
    main()
