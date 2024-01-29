from cProfile import run
import os
import time
import sys
from configs import *
from utils import *
from program_mgr import program_mgr
from OSML import OSML
from res.baseline.baseline import baseline
import subprocess
import shlex
import random
import numpy as np
import subprocess

random.seed(0)
nan=np.nan

def run_workload(workload,scheduler):
    print("workload:", workload)
    print("scheduler:", scheduler)
    #os.system("./reset.sh")
    write_workload(workload)
    if scheduler == "baseline":
        mgr = program_mgr(config_path=ROOT + "/workload.txt", regular_update=True, log_prefix="baseline", manage = False)
        baseline(mgr,terminate_when_QoS_is_met=True, terminate_when_timeout=True, timeout=200)
        mgr.end_all()
        log_path=mgr.end_log_thread()

    elif scheduler == "OSML":
        mgr = program_mgr(config_path=ROOT + "/workload.txt", regular_update=True, log_prefix="OSML")
        OSML(mgr,terminate_when_QoS_is_met=True, terminate_when_timeout=True, timeout=200)
        mgr.end_all()
        log_path=mgr.end_log_thread()

    elif scheduler == "PARTIES":
        os.chdir(ROOT+"/res/PARTIES/manager/") 
        p = os.system('python PARTIES.py')
        os.chdir(ROOT) 
        log_path="/home/OSML_Artifact/res/PARTIES/manager/log.txt"

    elif scheduler == "clite":
        os.chdir(ROOT+"/res/clite/")
        p = os.system('python clite.py')
        os.chdir(ROOT) 
        log_path="/home/OSML_Artifact/res/clite/log.txt"

    os.system("./reset.sh")
    time.sleep(2)
    return log_path

def write_workload(workloads):
    # for OSML and nopartition
    workload_str="PCT"
    for workload in workloads:
        workload_str+="\n"
        workload_str+=" ".join([str(item) for item in workload][:3])
    with open(ROOT+"/workload.txt","w") as f:
        f.write(workload_str)
    with open(ROOT+"/res/PARTIES/manager/workload.txt","w") as f:
        f.write(workload_str)
    with open(ROOT+"/res/clite/workload.txt","w") as f:
        f.write(workload_str)

def rename_log(path,new_path):
    os.system("mv {} {}".format(path,new_path))

def run_evaluation(workload,background,cover=True,schedulers=["clite"]):
    tail=time.strftime("%Y%m%d_%H%M%S", time.localtime())
    for scheduler in schedulers:
        print_color("workload:{}\nscheduler:{}".format(workload,scheduler))
        workload_name="workload_{}".format("+".join([item[0] for item in workload]))
        pcts=[w[1] for w in workload][:3]

        if len(background)>0:
            workload_name+="_background_{}".format("+".join([item[0] for item in background]))
        dir_name="evaluations/"+workload_name
        new_log_path="/{}_{}_{}.txt".format(scheduler,"+".join([w[0]+"@"+str(w[1]) for w in workload]),tail)
        os.system("mkdir -p {}".format(dir_name))
        case_name=dir_name+"/{}_{}".format(scheduler,"+".join([w[0]+"@"+str(w[1]) for w in workload]))

        log_path=run_workload(workload,scheduler)
        rename_log(log_path,dir_name+new_log_path)
        


def run_exps(scheduler):
    benchmarks=["nginx"]
    pcts=[30]

    workload=[]
    background=[]
    for _,pct in enumerate(pcts):
        workload.append([benchmarks[_],pct,36,"PCT"])
    write_workload(workload)

    tail=time.strftime("%Y%m%d_%H%M%S", time.localtime())
    workload_name="workload_{}".format("+".join([item for item in benchmarks]))
    new_log_path="/{}_{}_{}.txt".format(scheduler,"+".join([w[0]+"@"+str(w[1]) for w in zip(benchmarks,pcts)]),tail)
    log_path=run_workload(workload,scheduler)
    #os.system("cp {} {}".format(log_path,"/home/OSML_Artifact/experiments/exps/{}".format(workload_name)+new_log_path))
    print("log_path:", "/home/OSML_Artifact/experiments/exps/{}".format(workload_name)+new_log_path)


if __name__=="__main__":
    if len(sys.argv)==2:
        run_exps(sys.argv[1])
    else:
        print("Usage:\n python evaluation.py [baseline/OSML/PARTIES/clite]")
