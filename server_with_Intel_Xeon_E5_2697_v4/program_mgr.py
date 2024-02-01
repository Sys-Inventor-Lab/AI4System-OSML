import collections
import os
import re
import psutil
import subprocess
import threading
import numpy as np
import datetime
import time
import random
import pandas as pd
import warnings
import logging
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow.compat.v1 as tf
import hashlib
from subprocess import check_output
from itertools import product
from collections import deque, defaultdict
from copy import deepcopy
from configs import *
from utils import *

# ML models
from models.Model_A import Model_A
from models.Model_A_shadow import Model_A_shadow
from models.Model_B import Model_B
from models.Model_B_shadow import Model_B_shadow
from models.Model_C import Model_C

logger=logging.getLogger(__name__)

model_a = None
model_a_shadow = None
model_b = None
model_b_shadow = None
model_c_add = None
model_c_sub = None


def init_models():
    global model_a, model_a_shadow, model_b, model_b_shadow, model_c_add, model_c_sub
    model_a = Model_A()
    model_a_shadow = Model_A_shadow()
    model_b = Model_B()
    model_b_shadow = Model_B_shadow()
    with tf.variable_scope('add'):
        model_c_add = Model_C(len(ACTION_SPACE_ADD), len(
            C_FEATURES["s"]), "model_c_add", output_graph=False, e_greedy_increment=True)
    with tf.variable_scope('sub'):
        model_c_sub = Model_C(len(ACTION_SPACE_SUB), len(
            C_FEATURES["s"]), "model_c_sub", output_graph=False, e_greedy_increment=True)

# @timer


def filter_A_solutions(solutions, idle, mgr):
    sorted_keys = ["OAA", "RCliff"]
    candidates = {}
    for key in sorted_keys:
        if solutions[key]["cores"] <= idle["cores"] and solutions[key]["ways"] <= idle["ways"]:
            candidates[key] = solutions[key]
    return candidates


def sorted_B_solutions(priority):
    if priority == 0:
        solutions = ["Default_Spared"]
        dominated_solutions = [
            "Core_Dominated_Spared", "Cache_Dominated_Spared"]
        solutions.extend(random.choice(
            [dominated_solutions, dominated_solutions[::-1]]))
        return solutions
    if priority == 1:
        return ["Core_Dominated_Spared", "Default_Spared", "Cache_Dominated_Spared"]
    elif priority == 2:
        return ["Cache_Dominated_Spared", "Default_Spared", "Core_Dominated_Spared"]

# @timer


def filter_B_solutions(solutions, diffs):
    if not isinstance(diffs, dict):
        diffs = {_: item for _, item in enumerate(diffs)}
    keys = list(solutions.keys())
    candidates = {}

    def B_solution_dfs(key_idx):
        if diff["cores"] <= 0 and diff["ways"] <= 0:
            B_solutions.append(
                {keys[vidx]: solutions[keys[vidx]] for vidx in visited})
            return
        if len(visited) == MAX_INVOLVED:
            return
        for idx in range(key_idx + 1, len(keys)):
            if keys[idx][0] in [keys[vidx][0] for vidx in visited]:
                continue
            visited.append(idx)
            diff["cores"] -= solutions[keys[idx]]["cores"]
            diff["ways"] -= solutions[keys[idx]]["ways"]
            B_solution_dfs(idx)
            diff["cores"] += solutions[keys[idx]]["cores"]
            diff["ways"] += solutions[keys[idx]]["ways"]
            visited.remove(idx)

    for diff_key in diffs:
        diff = deepcopy(diffs[diff_key])
        B_solutions = []
        visited = []
        B_solution_dfs(0)
        B_solutions.sort(key=lambda x: sum([key[1] for key in x]))
        B_solutions.sort(key=lambda x: len(x), reverse=True)
        if len(B_solutions) > 0:
            candidates[diff_key] = B_solutions
    return candidates

# @timer


def filter_B_shadow_solutions(B_shadow_points, diffs, mgr):
    '''
    :param B_shadow_points:
    :param diffs:
    :param mgr:
    :return: {diff_idx:{}}
    '''
    if not AGGRESSIVE:
        for key in keys:
            if B_shadow_points[key]["QoS_Reduction"] > ACCEPTABLE_SLOWDOWN:
                B_shadow_points.pop(key)
    keys = list(B_shadow_points.keys())
    # print("B_shadow_points:", B_shadow_points)

    candidates = {}

    def B_shadow_solution_dfs(key_idx):
        if diff["cores"] <= 0 and diff["ways"] <= 0 and len(visited) > 0:
            B_shadow_solutions.append(
                {keys[vidx]: B_shadow_points[keys[vidx]] for vidx in visited})
            return
        if len(visited) == MAX_INVOLVED - 1:
            return
        for idx in range(key_idx, len(keys)):
            if keys[idx][0] in [keys[vidx][0] for vidx in visited]:
                continue
            if mgr.is_deprived(keys[idx][0]):
                continue
            visited.append(idx)
            diff["cores"] -= keys[idx][2]
            diff["ways"] -= keys[idx][1]
            B_shadow_solution_dfs(idx)
            diff["cores"] += keys[idx][2]
            diff["ways"] += keys[idx][1]
            visited.remove(idx)

    for diff_idx in diffs:
        diff = deepcopy(diffs[diff_idx])
        B_shadow_solutions = []
        visited = []
        B_shadow_solution_dfs(0)
        B_shadow_solutions.sort(key=lambda x: sum(
            [x[key]["QoS_Reduction"] for key in x]))  # Sort in order of QoS reduction
        # Sort in order of number of applications involved
        B_shadow_solutions.sort(key=lambda x: len(x))
        if len(B_shadow_solutions) > 0:
            candidates[diff_idx] = B_shadow_solutions
    return candidates

# @timer


def generate_deprivation_policy(solution, diff, mgr, app):
    diff = deepcopy(diff)
    deprivation_policy = {key: {"cores": 0, "ways": 0} for key in solution}
    aggressive = not mgr.is_QoS_met(app)
    # aggressive=AGGRESSIVE
    while (diff["cores"] > 0 or diff["ways"] > 0):
        success = []
        if aggressive:
            for key in solution:
                if diff["cores"] > 0 and deprivation_policy[key]["cores"] < solution[key]["cores"] and mgr.programs[key[0]].core_len-deprivation_policy[key]["cores"] > 1:
                    deprivation_policy[key]["cores"] += 1
                    diff["cores"] -= 1
                    success.append(True)
                if diff["ways"] > 0 and deprivation_policy[key]["ways"] < solution[key]["ways"] and mgr.programs[key[0]].way_len-deprivation_policy[key]["ways"] > 1:
                    deprivation_policy[key]["ways"] += 1
                    diff["ways"] -= 1
                    success.append(True)
        else:
            for key in solution:
                if diff["cores"] > 0 and deprivation_policy[key]["cores"] < solution[key]["cores"]\
                        and mgr.programs[key[0]].core_len-deprivation_policy[key]["cores"] > mgr.programs[key[0]].A_points["RCliff"]["cores"]:
                    deprivation_policy[key]["cores"] += 1
                    diff["cores"] -= 1
                    success.append(True)
                if diff["ways"] > 0 and deprivation_policy[key]["ways"] < solution[key]["ways"]\
                        and mgr.programs[key[0]].way_len-deprivation_policy[key]["ways"] > mgr.programs[key[0]].A_points["RCliff"]["ways"]:
                    deprivation_policy[key]["ways"] += 1
                    diff["ways"] -= 1
                    success.append(True)
        if len(success) == 0:
            break
    return deprivation_policy


class program_mgr:
    def __init__(self, config_path=None, regular_update=False, nopartition=False, log_prefix="OSML", manage=True, enable_models=False):
        self.pending_queue = collections.OrderedDict()
        self.programs = collections.OrderedDict()
        self.resource_used = {"cores": 0, "ways": 0}
        self.start_time = None
        self.QoS_met_time = None
        self.last_action = None
        self.config_path = config_path
        self.processed_config = set()
        self.regular_update = regular_update
        self.next_proc_id = 0
        self.available_COS_id = [i for i in range(1, N_COS)]
        self.proc_id_2_programs = {}
        self.sharing_policies = {}
        self.log = {"workload": [], "latency": [], "allocation": []}
        self.log_thread = None
        self.log_thread_configs = {"running": False}
        self.allocation_history = collections.deque(maxlen=5)
        self.revert_event = None
        self.deadlock = False
        self.manage = manage
        self.enable_models = enable_models
        if self.enable_models:
            init_models()
        self.update_pending_queue()
        self.log_path = ROOT+"/logs/"+log_prefix+"_"
        self.log_path += "+".join(["{}@{}".format(name, self.pending_queue[name].RPS) for name in self.pending_queue])+"_"
        self.log_path += time.strftime("%Y%m%d_%H%M%S",time.localtime())+".txt"
        self.TMP_DIR = ROOT + "/tmp/"
        self.monitor()

    def update_pending_queue(self):
        if self.config_path:
            with open(self.config_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    if line in self.processed_config:
                        continue
                    params = line.split()
                    for i in range(2, len(params)):
                        try:
                            params[i] = int(params[i])
                        except:
                            pass
                    if params[0] == "CHANGE_RPS":
                        name = params[1]
                        if name not in CHANGE_RPS_STR:
                            raise Exception(
                                "change_RPS does not support {}. It is only for applications in Tailbench.".format(name))
                        self.add_RPS_to_change(*params[1:])
                    else:
                        self.add_app(*params)
                    self.processed_config.add(line)


    def get_proc_id(self):
        proc_id = self.next_proc_id
        self.next_proc_id += 1
        return proc_id

    def get_COS_id(self):
        if len(self.available_COS_id) > 0:
            COS_id = self.available_COS_id[0]
            self.available_COS_id.remove(COS_id)
        else:
            COS_id = None
        return COS_id

    def put_COS_id(self, COS_id):
        self.available_COS_id.append(COS_id)
        self.available_COS_id.sort()

    def monitor(self):
        tail = "" if PQOS_OUTPUT_ENABLED else " 1>/dev/null"
        if not os.path.exists(self.TMP_DIR):
            os.makedirs(self.TMP_DIR)
        pqosout = open(self.TMP_DIR + "pqos.all", "w")
        subprocess.call(
            "pqos -I -i 1 -m all:0-{} &".format(N_CORES-1), shell=True, stdout=pqosout)

    def add_app(self, name, parse, value, n_threads=MAX_THREADS, launch_time=0, end_time=None):
        assert parse in ["RPS", "PCT"]
        if parse == "PCT":
            RPS = int(MAX_LOAD[name] * value / 100)
            RPS_str = "{}% max load".format(value)
        elif parse == "RPS":
            RPS = value
            RPS_str = str(value)

        p = program(name=name,
                    proc_id=self.get_proc_id(),
                    COS_id=self.get_COS_id(),
                    RPS=RPS,
                    RPS_str=RPS_str,
                    n_threads=n_threads,
                    launch_time=launch_time,
                    end_time=end_time,
                    regular_update=self.regular_update)
        self.pending_queue[name] = p
        self.proc_id_2_programs[p.proc_id] = p
        self.log["workload"].append({"name": name, "RPS": RPS, "n_threads": n_threads,
                                    "proc_id": p.proc_id, "launch_time": launch_time, "end_time": end_time})

    def add_RPS_to_change(self, name, parse, value, time):
        assert parse in ["RPS", "PCT"]
        if parse == "PCT":
            RPS = int(MAX_LOAD[name] * value / 100)
            RPS_str = "{}% max load".format(value)
        elif parse == "RPS":
            RPS = value
            RPS_str = str(value)
        self.pending_queue[name].add_RPS_to_change(RPS, RPS_str, time)

    def can_be_launched(self, name):
        if (name in self.pending_queue) and (self.start_time == None or time.time() - self.start_time >= self.pending_queue[name].launch_time):
            return True
        else:
            return False

    def can_be_ended(self, name):
        if (name in self.programs) and (self.programs[name].end_time is not None) and (time.time() - self.start_time >= self.programs[name].end_time):
            return True
        else:
            return False

    def all_done(self):
        return (len(self.pending_queue) == 0 and len(self.programs) == 0) or (len(self.programs) > 0 and all([self.programs[name].mode == MODE.Dead for name in self.programs]))

    def RPS_can_be_changed(self, name):
        if (name in self.programs) and (len(self.programs[name].RPS_to_change) > 0) and (time.time() - self.start_time >= self.programs[name].RPS_to_change[0]["time"]):
            return True
        else:
            return False

    def is_all_QoS_met(self):
        result = all([self.programs[name].is_QoS_met()
                     for name in self.programs]) and len(self.pending_queue) == 0
        if result and self.QoS_met_time is None:
            self.QoS_met_time = time.time()
        elif not result:
            self.QoS_met_time = None
        return result

    def is_deprived(self, name):
        if name not in self.programs:
            return None
        return self.programs[name].deprived

    def is_under_provision(self, name):
        if name not in self.programs:
            return None
        return self.programs[name].is_under_provision()

    def is_over_provision(self, name):
        if name not in self.programs:
            return None
        idle = self.resource_idle()
        if idle["cores"] > 0 and idle["ways"] > 0:
            return False
        else:
            return self.programs[name].is_over_provision()

    def get_QoS_violation(self, name):
        if name not in self.programs:
            return None
        QoS_violation = self.programs[name].get_QoS_violation()
        return QoS_violation

    def is_QoS_met(self, name):
        if name not in self.programs:
            return None
        return self.programs[name].is_QoS_met()

    def get_latency(self, name):
        if name not in self.programs:
            return None
        return self.programs[name].get_latency()

    def is_cores_under_provision(self, name):
        if name not in self.programs:
            return None
        return self.programs[name].is_cores_under_provision()

    def get_hungry_apps(self):
        return [app for app in self.programs if self.programs[app].mode == MODE.Hungry]

    def get_unmanaged_apps(self):
        return [app for app in self.programs if self.programs[app].mode == MODE.Unmanaged]

    def launch(self, name, core_len=None, way_len=None):
        # Launch an application and allocate the remaining resources on the platform to it .
        if not self.can_be_launched(name):
            return False

        self.programs[name] = self.pending_queue[name]
        self.pending_queue.pop(name)

        if not self.manage:
            self.programs[name].launch()
        else:
            if core_len is None:
                core_len = N_CORES - self.resource_used["cores"]
            if way_len is None:
                way_len = N_WAYS - self.resource_used["ways"]
            cores_required = 0 if core_len > 0 else 1
            ways_required = 0 if way_len > 0 else 1

            if not (cores_required == 0 and ways_required == 0):
                victim = self.select_victim(cores_required, ways_required)
                self.allocate_diff(
                    victim, {"cores": -1*cores_required, "ways": -1*ways_required}, propagate=True)
            self.programs[name].launch(core_start=self.resource_used["cores"],
                                       core_len=core_len+cores_required,
                                       way_start=self.resource_used["ways"],
                                       way_len=way_len+ways_required)
            self.resource_used["cores"] += core_len
            self.resource_used["ways"] += way_len
            self.propagate_allocation()

        if self.start_time is None:
            self.start_time = time.time()

        return True

    def launch_all(self):
        for name in list(self.pending_queue.keys()):
            if not self.can_be_launched(name):
                continue
            self.programs[name] = self.pending_queue[name]
            self.pending_queue.pop(name)
            self.programs[name].launch(core_start=0, core_len=None,
                                       way_start=0, way_len=None)
            if self.start_time is None:
                self.start_time = time.time()
        return True

    def start_log_thread(self, interval=1):
        self.log_thread_configs["running"] = True
        self.log_thread_configs["interval"] = 1
        self.log_thread = threading.Thread(
            target=self.regular_log, args=(self.log_thread_configs,), daemon=True)
        self.log_thread.start()

    def end_log_thread(self):
        os.system("mkdir -p {}".format(ROOT+"/logs/"))
        with open(self.log_path, "w") as f:
            f.write(str(self.log))
        os.system("cp {} {}".format(self.log_path,
                  ROOT+"/logs/OSML_log_latest.txt"))
        self.log_thread_configs["running"] = False
        if self.log_thread is not None:
            self.log_thread.join()
        return self.log_path

    def neighbors(self, name):
        neighbors = list(self.programs.keys())
        neighbors.remove(name)
        return neighbors

    def candidates_for_model_B(self, name):
        candidates = list(self.programs.keys())
        to_remove = set()
        to_remove.add(name)
        for app in candidates:
            if not self.programs[app].acceptable_QoS_slowdown == 0:
                to_remove.add(app)
            if not AGGRESSIVE and not self.is_QoS_met(app):
                to_remove.add(app)
            if not self.programs[app].mode == MODE.Managed:
                to_remove.add(app)
        for app in to_remove:
            candidates.remove(app)
        return candidates

    def get_under_provision_apps(self):
        under_provision_apps = [app for app in list(
            self.programs) if self.is_under_provision(app)]
        return under_provision_apps

    def get_over_provision_apps(self):
        over_provision_apps = [app for app in list(
            self.programs) if self.is_over_provision(app)]
        if self.deadlock and len(over_provision_apps) == 0:
            for app in sorted([app for app in list(self.programs)], key=lambda x: self.get_QoS_violation(x)):
                if (self.programs[app].core_len is not None and self.programs[app].core_len > 1) or (self.programs[app].way_len is not None and self.programs[app].way_len > 1):
                    over_provision_apps.append(app)
                    break
            self.deadlock = False
        return over_provision_apps

    def select_victim(self, cores_required, ways_required, exclude=[]):
        sorted_programs = sorted(self.programs.keys(),
                                 key=lambda x: self.get_QoS_violation(x))
        for name in sorted_programs:
            if name in exclude:
                continue
            if self.programs[name].core_len-cores_required > 0 and self.programs[name].way_len-ways_required > 0:
                return name
        return None

    def end(self, name):
        if not self.can_be_ended(name):
            return False
        else:
            self.put_COS_id(self.programs[name].COS_id)
            self.programs[name].end()
            for proc_id_str in self.sharing_policies:
                if str(self.programs[name].proc_id) in proc_id_str:
                    self.sharing_policies.pop(proc_id_str)
            self.programs.pop(name)
            self.propagate_allocation()

    def change_RPS(self, name):
        if not self.RPS_can_be_changed(name):
            return False
        else:
            self.programs[name].change_RPS()
            

    def end_all(self):
        for name in list(self.programs.keys()):
            self.end(name)

    def register_revert_event(self, names, callback=None, args=None):
        if isinstance(names, str):
            names = [names]
        self.revert_event = {"names": names,
                             "callback": callback, "args": args}

    def check_revert_event(self):
        if self.revert_event is None:
            return
        if any([not self.is_QoS_met(name) for name in self.revert_event["names"]]):
            # QoS violation happens
            #print_color("Revert because the QoS of {} is not met.".format(self.revert_event["names"]), "red")
            self.revert()
            if self.revert_event["callback"] is not None:
                self.revert_event["callback"](self.revert_event["args"])
        self.revert_event = None

    def use_model_A(self, name):
        features = self.get_features(name, A_FEATURES)
        output = model_a.use_model(features)
        if MBA_SUPPORT:
            result = {"RCliff": {"ways": cache_2_way(output[0], MB_PER_WAY), "cores": int(round(output[1]))},
                      "OAA": {"ways": cache_2_way(output[2], MB_PER_WAY), "cores": int(round(output[3]))},
                      "OAA_Bandwidth": output[4]}
        else:
            result = {"RCliff": {"ways": cache_2_way(output[0], MB_PER_WAY), "cores": int(round(output[1]))},
                      "OAA": {"ways": cache_2_way(output[2], MB_PER_WAY), "cores": int(round(output[3]))}}
        print_color("==> Use Model A for {}.".format(name), "cyan")
        logger.info("Use Model A for {}.".format(name))
        # print_color("Input:{}".format(features),"cyan")
        # print_color("Output:{}".format(output), "cyan")
        # print_color("Trimmed_result:{}".format(result), "cyan")

        for key in ["RCliff", "OAA"]:
            result[key]["ways"] = min(N_WAYS, result[key]["ways"])
            result[key]["ways"] = max(1, result[key]["ways"])
            result[key]["cores"] = min(N_CORES, result[key]["cores"])
            result[key]["cores"] = max(1, result[key]["cores"])
        self.programs[name].A_points = deepcopy(result)
        return result

    def use_model_A_shadow(self, name):
        A_features = self.get_features(name, A_FEATURES)
        neighbor_features = self.get_neighbor_features(name)
        features = np.concatenate((A_features, neighbor_features), axis=0)
        output = model_a_shadow.use_model(features)
        if MBA_SUPPORT:
            result = {"RCliff": {"ways": cache_2_way(output[0], MB_PER_WAY), "cores": int(round(output[1]))},
                      "OAA": {"ways": cache_2_way(output[2], MB_PER_WAY), "cores": int(round(output[3]))},
                      "OAA_Bandwidth": output[4]}
        else:
            result = {"RCliff": {"ways": cache_2_way(output[0], MB_PER_WAY), "cores": int(round(output[1]))},
                      "OAA": {"ways": cache_2_way(output[2], MB_PER_WAY), "cores": int(round(output[3]))}}
        print_color("==> Use Model A' for {}.".format(name), "cyan")
        logger.info("Use Model A' for {}.".format(name))
        # print_color("Output:{}".format(output), "cyan")

        for key in ["RCliff", "OAA"]:
            result[key]["ways"] = min(N_WAYS, result[key]["ways"])
            result[key]["ways"] = max(1, result[key]["ways"])
            result[key]["cores"] = min(N_CORES, result[key]["cores"])
            result[key]["cores"] = max(1, result[key]["cores"])

        # print_color("Trimmed_result:{}".format(result), "cyan")

        self.programs[name].A_points = deepcopy(result)
        return result

    def use_model_B(self, names, QoS_Reductions):
        if not isinstance(names, list):
            names = [names]
        if not isinstance(QoS_Reductions, list):
            QoS_Reductions = [QoS_Reductions]
        B_points = {}
        colocated_features = self.get_colocated_features(names)
        print_color("==> Use Model B.", "cyan")
        logger.info("Use Model B.")
        for QoS_Reduction in QoS_Reductions:
            B_features = {}
            for name in names:
                B_features[name] = colocated_features[name][:-1]
                B_features[name].append(QoS_Reduction)
                output = model_b.use_model(B_features[name])
                B_points[(name, QoS_Reduction, "Core_Dominated_Spared")] = {"ways": cache_2_way(output[0], MB_PER_WAY),
                                                                            "cores": int(round(output[1]))}
                B_points[(name, QoS_Reduction, "Default_Spared")] = {"ways": cache_2_way(output[2], MB_PER_WAY),
                                                                     "cores": int(round(output[3]))}
                B_points[(name, QoS_Reduction, "Cache_Dominated_Spared")] = {"ways": cache_2_way(output[4], MB_PER_WAY),
                                                                             "cores": int(round(output[5]))}
                # print_color("----------\nName:{}".format(name),"cyan")
                # print_color("QoS_Reduction:{}".format(QoS_Reduction),"cyan")
                # print_color("Input:{}".format(B_features[name]),"cyan")
                # print_color("Output:{}".format(output),"cyan")
                # print_color("B_points:{}; {}; {}".format(B_points[(name, QoS_Reduction, "Core_Dominated_Spared")],B_points[(name, QoS_Reduction, "Default_Spared")],B_points[(name, QoS_Reduction, "Cache_Dominated_Spared")]),"cyan")

        return B_points

    def use_model_B_shadow(self, names, sharing_policies):
        '''
        Call Model B' to predict QoS slowdown after resource deprivation
        :param names: Apps whose resources are to be deprived
        :param sharing_policies: Amount of resources needed
        :return: {(name,cache ways needed,cores needed):{"QoS_Reduction":0.05,"ways":cache ways needed,"cores":cores needed}}
        '''
        if not isinstance(names, list):
            names = [names]
        if not isinstance(sharing_policies, list):
            sharing_policies = [sharing_policies]
        B_shadow_points = {}
        colocated_features = self.get_colocated_features(names)
        for sharing_policy in sharing_policies:
            B_shadow_features = {}
            for name in names:
                B_shadow_features[name] = colocated_features[name][:-1]
                target_cache = colocated_features[name][9] - \
                    sharing_policy["ways"] * MB_PER_WAY
                target_core = colocated_features[name][10] - \
                    sharing_policy["cores"]
                if target_cache <= 0 or target_core <= 0:
                    continue
                B_shadow_features[name].extend([target_cache, target_core])
                output = model_b_shadow.use_model(B_shadow_features[name])
                B_shadow_points[(name, sharing_policy["ways"], sharing_policy["cores"])] = {"QoS_Reduction": output[0],
                                                                                            "ways": sharing_policy["ways"],
                                                                                            "cores": sharing_policy["cores"]}
        return B_shadow_points

    def model_C_reward(self, latency_t1, latency_t2, action):
        delta_latency = latency_t2-latency_t1
        if delta_latency > 0:
            reward = -np.log(1+delta_latency) - \
                (action["cores"]+action["ways"])
        elif delta_latency < 0:
            reward = np.log(1-delta_latency) - (action["cores"]+action["ways"])
        else:
            reward = - (action["cores"]+action["ways"])
        return reward

    def use_model_C_add(self, name):
        cores_under_provision = self.programs[name].is_cores_under_provision()
        latency_t1 = self.get_features(name, "Latency")[0]
        features = self.get_features(name, C_FEATURES["s"])
        predicted_action_id = model_c_add.choose_action(features)
        predicted_action = ACTION_SPACE_ADD[predicted_action_id]
        idle = self.resource_idle()
        action = {"cores": predicted_action[0], "ways": predicted_action[1]}

        self.last_action = {"model": "model_C_add",
                            "target": name,
                            "latency_t1": latency_t1,
                            "features": features,
                            }
        if idle["cores"] > 0 or idle["ways"] > 0:
            action["cores"] = min(idle["cores"], action["cores"])
            action["ways"] = min(idle["ways"], action["ways"])
            action["cores"] = max(-max(0,
                                  (self.programs[name].core_len-1)), action["cores"])
            action["ways"] = max(-max(0,
                                 (self.programs[name].way_len-1)), action["ways"])
            if cores_under_provision:
                action["cores"] = max(0, action["cores"])
            else:
                action["ways"] = max(0, action["ways"])
            if (action["cores"], action["ways"]) in ACTION_SPACE_SUB:
                action["cores"] = max(action["cores"], 0)
                action["ways"] = max(action["ways"], 0)
            if not action["cores"]==0 and action["ways"]==0:    
                print_color("==> Model_C_add, allocate {} to {}".format(str(action), name), "green")
                logger.info("Model_C_add, allocate {} to {}".format(str(action), name))
            self.allocate_diff(name, action, propagate=True)
            self.last_action["action"] = action
        elif SHARING:
            # print_color("Idle resource is not enough. Try resource sharing among applications.", "red")
            diff = {"cores": max(0, predicted_action[0]-idle["cores"]),
                    "ways": max(0, predicted_action[1]-idle["ways"])}
            sharing_policy = {"cores": max(0, predicted_action[0]),
                              "ways": max(0, predicted_action[1])}
            B_shadow_points = self.use_model_B_shadow(
                self.candidates_for_model_B(name), sharing_policy)
            B_shadow_solutions = filter_B_shadow_solutions(
                B_shadow_points, {0: diff}, self)
            # print_color("B_shadow_solutions: {}".format(B_shadow_solutions), "cyan")

            if len(B_shadow_solutions) == 0:
                self.deadlock = True
                # print_color("Resource not enough. B_shadow_solutions is empty.", "red")
            else:
                for case in B_shadow_solutions:
                    B_shadow_solution = B_shadow_solutions[case][0]
                    deprivation_policy = generate_deprivation_policy(
                        B_shadow_solution, diff, self, name)
                    # print_color("Deprivation policy: {}".format(deprivation_policy), "cyan")
                    apps = [key[0] for key in deprivation_policy.keys()]
                    action_cores = sum([deprivation_policy[key]["cores"]
                                       for key in deprivation_policy])
                    action_ways = sum([deprivation_policy[key]["ways"]
                                      for key in deprivation_policy])
                    if action_cores == 0 and action_ways == 0:
                        continue
                    for key in deprivation_policy:
                        self.allocate_diff(key[0], {"cores": -deprivation_policy[key]["cores"],
                                                    "ways": -deprivation_policy[key]["ways"]}, propagate=False)
                    apps.append(name)
                    self.allocate_sharing(
                        apps, {"cores": action_cores, "ways": action_ways}, propagate=True)
                    self.last_action["action"] = {
                        "cores": action_cores, "ways": action_ways}
                    # print_color("Enable resource sharing.", "green")
                    break
        else:
            self.deadlock = True
            # print_color("Resource not enough.", "red")

    def revert_partially(self, args):
        cores_under_provision = self.is_cores_under_provision(args[0])

        if cores_under_provision:
            action = {"cores": 0, "ways": args[1]["ways"]}
        else:
            action = {"cores": args[1]["cores"], "ways": 0}

        self.allocate_diff(args[0], action, propagate=True)
        self.last_action["action"] = action

    def use_model_C_sub(self, name):
        latency_t1 = self.get_features(name, "Latency")[0]
        features = self.get_features(name, C_FEATURES["s"])
        predicted_action_id = model_c_sub.choose_action(features)
        predicted_action = ACTION_SPACE_SUB[predicted_action_id]
        action = {"cores": 0, "ways": 0}
        action["cores"] = min(0, predicted_action[0])
        action["ways"] = min(0, predicted_action[1])
        action["cores"] = max(-max(0,
                              (self.programs[name].core_len-1)), action["cores"])
        action["ways"] = max(-max(0,
                             (self.programs[name].way_len-1)), action["ways"])
        
        if not action["cores"]==0 and action["ways"]==0:    
            print_color("==> Model_C_sub, allocate {} to {}".format(str(action), name), "green")
            logger.info("Model_C_sub, allocate {} to {}".format(str(action), name))
        self.allocate_diff(name, action, propagate=True)
        self.register_revert_event(name, callback=self.revert_partially, args=(name, action))
        self.last_action = {"model": "model_C_sub",
                            "target": name,
                            "latency_t1": latency_t1,
                            "features": features,
                            "action": action}

    def process_last_model_C_action(self):
        if self.last_action is None:
            return

        if "action" not in self.last_action:
            self.last_action = None
            return

        model = self.last_action["model"]
        name = self.last_action["target"]
        features = self.last_action["features"]
        action = self.last_action["action"]
        latency_t1 = self.last_action["latency_t1"]
        latency_t2 = self.get_features(name, "Latency")[0]
        reward = self.model_C_reward(latency_t1, latency_t2, action)
        # print_color("reward:{}".format(reward), "cyan")
        features_ = self.get_features(name, C_FEATURES["s"])

        if model == "model_C_add":
            model_c = model_c_add
            action_id = ACTION_ID_ADD[(action["cores"], action["ways"])]

        elif model == "model_C_sub":
            model_c = model_c_sub
            action_id = ACTION_ID_SUB[(action["cores"], action["ways"])]

        model_c.store_transition(features, action_id, reward, features_, True)
        model_c.learn()
        model_c.save()

        self.last_action = None

    def revert(self):
        if len(self.allocation_history) <= 1:
            # print_color("Revert failed, there are no records of previously executed actions.")
            return
        else:
            self.allocation_history.pop()
            current_allocation = self.allocation_history[-1]
            self.sharing_policies = deepcopy(
                current_allocation["allocation_sharing"])

            for name, allocation in current_allocation["allocation_independent"].items():
                if name not in self.programs:
                    return
                self.programs[name].core_start = allocation["core_start"]
                self.programs[name].core_len = allocation["core_len"]
                self.programs[name].way_start = allocation["way_start"]
                self.programs[name].way_len = allocation["way_len"]

            self.resource_used = deepcopy(current_allocation["resource_used"])

            for name, mode in current_allocation["mode"].items():
                self.programs[name].mode = mode

            for name, acceptable_QoS_slowdown in current_allocation["acceptable_QoS_slowdown"].items():
                self.programs[name].acceptable_QoS_slowdown = acceptable_QoS_slowdown

            self.conduct_allocation()

    def get_mode(self, name):
        return self.programs[name].get_mode()

    # @timer
    def get_features(self, names, keys):
        return_list = True
        if not isinstance(names, list):
            return_list = False
            names = [names]
        features = {}
        threads = []
        for name in names:
            thread = threading.Thread(
                target=self.get_features_for_threading, args=(name, features, keys))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        if return_list:
            return features
        else:
            return features[names[0]]

    def get_features_for_threading(self, name, features, keys):
        try:
            features[name] = self.programs[name].get_features(keys)
        except Exception as e:
            raise e
            pass

    def get_neighbor_features(self, name):
        neighbor_names = self.neighbors(name)
        neighbor_proc_ids = [
            self.programs[app].proc_id for app in neighbor_names]
        features = self.get_features(neighbor_names, ["MBL"])
        MBLs = [features[app][0] for app in features]
        Allocated_Cache = 0
        Allocated_Core = 0
        for proc_id_str in self.sharing_policies:
            proc_ids = [int(i) for i in proc_id_str.split("+")]
            if any([proc_id in neighbor_proc_ids for proc_id in proc_ids]):
                Allocated_Cache += way_2_cache(self.sharing_policies[proc_id_str]["allocation"]["ways"], MB_PER_WAY)
                Allocated_Core += self.sharing_policies[proc_id_str]["allocation"]["cores"]
        for app in neighbor_names:
            if not self.programs[app].way_len is None:
                Allocated_Cache += way_2_cache(self.programs[app].way_len, MB_PER_WAY)
            if not self.programs[app].core_len is None:
                Allocated_Core += self.programs[app].core_len
        arr = [round(sum(MBLs) / len(MBLs), 2) if len(MBLs) >=
               1 else 0, Allocated_Cache, Allocated_Core]
        return arr

    def get_colocated_features(self, names):
        return_dict = True
        if not isinstance(names, list):
            return_dict = False
            names = [names]
        colocated_features = {}
        features = self.get_features(names, COLLECT_FEATURES)
        N_features = {name: self.get_neighbor_features(name) for name in names}
        for name in names:
            colocated_features[name] = []
            colocated_features[name].extend(features[name][:-1])
            colocated_features[name].extend(N_features[name])
            colocated_features[name].append(features[name][-1])
        if return_dict:
            return colocated_features
        else:
            return list(colocated_features.values())[0]

    def regular_log(self, configs):
        while configs["running"]:
            time_remaining = configs["interval"] - \
                time.time() % configs["interval"]
            time.sleep(time_remaining)
            time_stamp = time.time()
            self.log_latency(time_stamp)
            self.log_allocation(time_stamp)

    def log_latency(self, time_stamp):
        latency_log = {"time": time_stamp, "latency": {}}
        names = list(self.programs.keys())
        for name in names:
            if self.programs[name].pid is None:
                continue
            latency = self.programs[name].get_latency()
            latency_log["latency"][name] = latency
        self.log["latency"].append(latency_log)

    def log_allocation(self, time_stamp):
        allocation_log = {"time": time_stamp,
                          "allocation_sharing": {}, "allocation_independent": {}}
        core_start = 0
        way_start = 0
        for _, proc_id_str in enumerate(self.sharing_policies):
            proc_ids = [int(i) for i in proc_id_str.split('+')]
            allocation_log["allocation_sharing"][proc_id_str] = {"apps": [self.proc_id_2_programs[proc_id].name for proc_id in proc_ids],
                                                                 "core_start": core_start,
                                                                 "core_len": self.sharing_policies[proc_id_str]["allocation"]["cores"],
                                                                 "way_start": way_start,
                                                                 "way_len": self.sharing_policies[proc_id_str]["allocation"]["ways"]}
            core_start += self.sharing_policies[proc_id_str]["allocation"]["cores"]
            way_start += self.sharing_policies[proc_id_str]["allocation"]["ways"]
        names = list(self.programs.keys())
        for name in names:
            if not (self.programs[name].core_start is None or self.programs[name].core_len is None):
                allocation_log["allocation_independent"][name] = {"core_start": self.programs[name].core_start,
                                                                  "core_len": self.programs[name].core_len,
                                                                  "way_start": self.programs[name].way_start,
                                                                  "way_len": self.programs[name].way_len}

        self.log["allocation"].append(allocation_log)

    def report_latency(self, names=None):
        if names is None:
            names = list(self.programs.keys())
        if type(names) == str:
            names = [names]
        print("")
        print("{} seconds".format(round(time.time()-self.start_time,2)))
        print("===============Latency===============")
        log_str=[]
        for name in names:
            latency = self.programs[name].get_features("Latency")[0]
            RPS_str = self.programs[name].RPS_str
            if latency <= QOS_TARGET[name]:
                met = "\033[32mQoS met\033[0m"
            elif latency/QOS_TARGET[name]<10:
                met = "\033[36m{}X QoS target\033[0m".format(round(latency/QOS_TARGET[name], 2))
            else:
                met = "\033[31m{}X QoS target\033[0m".format(round(latency/QOS_TARGET[name], 2))
            print("{}\t- {}\t- {}\t- {}".format(name, latency, RPS_str, met))
            log_str.append("{} - {} - {} - {}".format(name, latency, RPS_str, met))
        logger.info("Latency: {}".format("; ".join(log_str)))

    def report_allocation(self, names=None):
        if names is None:
            names = list(self.programs.keys())
        if type(names) == str:
            names = [names]
        log_str=[]
        core_start = 0
        way_start = 0
        if self.manage:
            print("===============Allocation===============")
            core_ranges = []
            way_ranges = []
            core_labels = []
            way_labels = []
            for _, proc_id_str in enumerate(self.sharing_policies):
                proc_ids = [int(i) for i in proc_id_str.split('+')]
                proc_names = [self.proc_id_2_programs[proc_id].name for proc_id in proc_ids]
                s="Sharing policy {}: {} sharing: ".format(_, ", ".join(proc_names))
                if self.sharing_policies[proc_id_str]["allocation"]["cores"] > 0:
                    core_end = core_start + self.sharing_policies[proc_id_str]["allocation"]["cores"] - 1
                    s += "{}-{} cores; ".format(core_start, core_end)
                    core_start += self.sharing_policies[proc_id_str]["allocation"]["cores"]
                if self.sharing_policies[proc_id_str]["allocation"]["ways"] > 0:
                    way_end = way_start + self.sharing_policies[proc_id_str]["allocation"]["ways"] - 1
                    s += "{}-{} ways; ".format(way_start, way_end)
                    way_start += self.sharing_policies[proc_id_str]["allocation"]["ways"]
                print(s)
                log_str.append(s)
                core_ranges.append(self.sharing_policies[proc_id_str]["allocation"]["cores"])
                way_ranges.append(self.sharing_policies[proc_id_str]["allocation"]["ways"])
                core_labels.append("+".join(proc_names))
                way_labels.append("+".join(proc_names))

        for name in names:
            state = ""
            if not (self.programs[name].core_start is None or self.programs[name].core_len is None):
                state += "{} cores ({}-{}); ".format(
                    self.programs[name].core_len,
                    self.programs[name].core_start,
                    self.programs[name].core_start + self.programs[name].core_len - 1)
            if not (self.programs[name].way_start is None or self.programs[name].way_len is None):
                state += "{} ways ({}-{}); ".format(
                    self.programs[name].way_len,
                    self.programs[name].way_start,
                    self.programs[name].way_start + self.programs[name].way_len - 1)
            if not state == "":
                s = name + " is allocated to: " + state
                core_ranges.append(self.programs[name].core_len)
                way_ranges.append(self.programs[name].way_len)
                core_labels.append(name)
                way_labels.append(name)
                log_str.append(s)
                print(s)

        if sum(core_ranges)<N_CORES:
            core_ranges.append(N_CORES-sum(core_ranges))
            core_labels.append("idle")
        if sum(way_ranges)<N_WAYS:
            way_ranges.append(N_WAYS-sum(way_ranges))
            way_labels.append("idle")

        print("Cores Allocation")        
        draw_bar_chart(N_CORES, core_ranges, core_labels)
        print("LLC Ways Allocation")
        draw_bar_chart(N_WAYS, way_ranges, way_labels)
        logger.info("Allocation: "+"; ".join(log_str))

    def size_pending(self):
        return len(self.pending_queue)

    def size_onfly(self):
        return len(self.programs)

    def size(self):
        return self.size_pending() + self.size_onfly()

    def resource_idle(self, exclude=None):
        if exclude is not None and self.get_mode(exclude) == MODE.Managed:
            return {"cores": N_CORES - self.resource_used["cores"] + self.programs[exclude].core_len,
                    "ways": N_WAYS - self.resource_used["ways"] + self.programs[exclude].way_len}
        else:
            return {"cores": N_CORES - self.resource_used["cores"], "ways": N_WAYS - self.resource_used["ways"]}

    def allocate(self, name, allocation, propagate=True):
        self.programs[name].core_len = int(allocation["cores"])
        self.programs[name].way_len = int(allocation["ways"])
        if propagate:
            self.propagate_allocation()

    #@timer
    def allocate_diff(self, name, allocation_diff, propagate=True):
        if name not in self.programs:
            self.propagate_allocation()
            return
        if allocation_diff["cores"] == 0 and allocation_diff["ways"] == 0:
            return
        self.programs[name].core_len += int(allocation_diff["cores"])
        self.programs[name].way_len += int(allocation_diff["ways"])
        if propagate:
            self.propagate_allocation()
        else:
            self.resource_used["cores"] += int(allocation_diff["cores"])
            self.resource_used["ways"] += int(allocation_diff["ways"])

    def allocate_sharing(self, names, allocation, propagate=True):
        '''
        Create a sharing policy.
        :param names: List of applications.
        :param allocation: Resources shared.
        :return: None
        '''
        proc_id_str = "+".join([str(proc_id) for proc_id in sorted(
            [self.programs[name].proc_id for name in names])])
        sharing_policy = {"allocation": allocation,
                          "COS_id": self.get_COS_id()}
        self.sharing_policies[proc_id_str] = sharing_policy
        for name in names:
            if name in self.programs:
                self.programs[name].sharing_policies[proc_id_str] = sharing_policy
        if propagate:
            self.propagate_allocation()

    def remove_sharing_policy(self, names):
        proc_id_str = "+".join([str(proc_id) for proc_id in sorted(
            [self.programs[name].proc_id for name in names])])
        if proc_id_str in self.sharing_policies:
            self.put_COS_id(self.sharing_policies[proc_id_str]["COS_id"])
            self.sharing_policies.pop(proc_id_str)
            for name in names:
                if name in self.programs:
                    self.programs[name].sharing_policies.pop(proc_id_str)

    def propagate_allocation(self):
        core_start = 0
        way_start = 0

        for proc_id_str in self.sharing_policies:
            self.sharing_policies[proc_id_str]["position"] = {}
            self.sharing_policies[proc_id_str]["position"]["core_start"] = core_start
            self.sharing_policies[proc_id_str]["position"]["core_len"] = self.sharing_policies[proc_id_str]["allocation"]["cores"]
            self.sharing_policies[proc_id_str]["position"]["way_start"] = way_start
            self.sharing_policies[proc_id_str]["position"]["way_len"] = self.sharing_policies[proc_id_str]["allocation"]["ways"]
            core_start += self.sharing_policies[proc_id_str]["allocation"]["cores"]
            way_start += self.sharing_policies[proc_id_str]["allocation"]["ways"]

        for n, p in self.programs.items():
            if p.core_len:
                assert (p.core_len >= 0)
                p.core_start = core_start
                core_start += p.core_len
            if p.way_len:
                assert (p.way_len >= 0)
                p.way_start = way_start
                way_start += p.way_len

        assert (core_start >= 0 and core_start <= N_CORES)
        assert (way_start >= 0 and way_start <= N_WAYS)
        self.resource_used["cores"] = core_start
        self.resource_used["ways"] = way_start

        for n, p in self.programs.items():
            if p.core_len and p.way_len and (p.mode == MODE.Unmanaged or p.mode == MODE.Hungry):
                p.mode = MODE.Managed

        self.allocation_history.append(
            {"allocation_sharing": deepcopy(self.sharing_policies),
             "allocation_independent": {n: {"core_start": p.core_start, "core_len": p.core_len, "way_start": p.way_start, "way_len": p.way_len} for n, p in self.programs.items()},
             "resource_used": deepcopy(self.resource_used),
             "mode": {n: p.mode for n, p in self.programs.items()},
             "acceptable_QoS_slowdown": {n: p.acceptable_QoS_slowdown for n, p in self.programs.items()}})

        self.conduct_allocation()

    def get_way_mask(self, way_list):
        bin_arr = [str(0)] * N_WAYS
        for way_idx in way_list:
            bin_arr[way_idx] = "1"
        bin_str = "".join(bin_arr)[::-1]
        return format(int(bin_str, 2), "x").rjust(5, "0").upper()

    def get_core_mask(self, core_list):
        return ",".join([str(core_idx) for core_idx in core_list])

    def conduct_allocation(self):
        tail = "" if PQOS_OUTPUT_ENABLED else " 1>/dev/null"

        def get_physical_cores(cores):
            assert type(cores) == list
            return [PHYSICAL_CORES[core_idx] for core_idx in cores]

        def allocate_core(p, core_used):
            try:
                if len(core_used) == 0:
                    return
                ps = psutil.Process(p.pid)
                ps_threads = ps.threads()
                for t in ps_threads:
                    os.sched_setaffinity(t.id, core_used)
            except:
                pass

        def allocate_llc(COS_id, way_used, core_used):
            try:
                if len(way_used) == 0 or len(core_used) == 0:
                    return
                way_used_str = self.get_way_mask(way_used)
                core_used_str = self.get_core_mask(core_used)
                subprocess.call("pqos --iface=msr -a llc:{}={}".format(COS_id, core_used_str) + tail, shell=True)
                subprocess.call("pqos --iface=msr -e llc:{}=0x{}".format(COS_id, way_used_str) + tail, shell=True)
            except Exception as e:
                raise e

        core_used = {name: [] for name in self.programs}
        core_independent = {name: [] for name in self.programs}
        core_shared = {proc_id_str: []
                       for proc_id_str in self.sharing_policies}

        way_independent = {name: [] for name in self.programs}
        way_shared = {proc_id_str: [] for proc_id_str in self.sharing_policies}

        for proc_id_str, policy in self.sharing_policies.items():
            proc_ids = [int(i) for i in proc_id_str.split('+')]
            core_shared[proc_id_str].extend(get_physical_cores(list(range(policy["position"]["core_start"],
                                                                          policy["position"]["core_start"] + policy["position"]["core_len"]))))
            way_shared[proc_id_str].extend(list(range(policy["position"]["way_start"],
                                                      policy["position"]["way_start"] + policy["position"]["way_len"])))
            for proc_id in proc_ids:
                p = self.proc_id_2_programs[proc_id]
                if p.mode != MODE.Dead:
                    core_used[self.proc_id_2_programs[proc_id].name].extend(get_physical_cores(list(range(policy["position"]["core_start"],
                                                                                                          policy["position"]["core_start"] +
                                                                                                          policy["position"]["core_len"]))))

        for name, p in self.programs.items():
            if not (p.core_start is None or p.core_len is None):
                core_used[name].extend(get_physical_cores(
                    list(range(p.core_start, p.core_start + p.core_len))))
                core_independent[name].extend(get_physical_cores(
                    list(range(p.core_start, p.core_start + p.core_len))))
            if not (p.way_start is None or p.way_len is None):
                way_independent[name].extend(
                    list(range(p.way_start, p.way_start + p.way_len)))

        # Allocate Core
        for name, p in self.programs.items():
            allocate_core(p, core_used[name])

        # Allocate LLC ways
        threads = []
        for proc_id_str in self.sharing_policies:
            thread = threading.Thread(target=allocate_llc, args=(
                self.sharing_policies[proc_id_str]["COS_id"], way_shared[proc_id_str], core_shared[proc_id_str]))
            thread.start()
            threads.append(thread)
        for name, p in self.programs.items():
            thread = threading.Thread(target=allocate_llc, args=(
                p.COS_id, way_independent[name], core_independent[name]))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.log_allocation(time.time())


class program:
    def __init__(self, name, proc_id, COS_id, RPS, RPS_str, n_threads, launch_time, end_time, regular_update):
        self.name = name
        self.pid = None
        self.pid_str = None
        self.pid_str_md5 = None
        self.proc_id = proc_id
        self.COS_id = COS_id
        self.RPS = RPS
        self.RPS_to_change = []
        self.RPS_to_change_str = []
        self.RPS_str = RPS_str
        self.n_threads = n_threads
        self.launch_time = launch_time
        self.end_time = end_time
        self.acceptable_QoS_slowdown = 0
        self.launched = False
        self.deprived = False

        self.mode = MODE.Unlaunched  # the application status

        self.last_latency = None
        self.last_IPS = None
        self.regular_update = regular_update
        self.history_latency = deque(maxlen=HISTORY_LEN)
        self.history_IPS = deque(maxlen=HISTORY_LEN)

        self.core_start = None
        self.core_len = None
        self.way_start = None
        self.way_len = None
        self.sharing_policies = {}
        self.state = {}

        self.A_points = {}

        self.TMP_DIR = ROOT + "/tmp/"

    def launch(self, core_start=None, core_len=None, way_start=None, way_len=None):
        self.core_start = core_start
        self.core_len = core_len
        self.way_start = way_start
        self.way_len = way_len
        if not (self.core_start is None or self.core_len is None):
            taskset_str = "taskset -c {}-{} ".format(
                self.core_start, self.core_start + self.core_len - 1)
        else:
            taskset_str = ""
        if isinstance(LAUNCH_STR[self.name], str):
            cmds = [LAUNCH_STR[self.name]]
        else:
            cmds = LAUNCH_STR[self.name]

        for cmd in cmds:
            if self.name in ["nodejs"]:
                time.sleep(3)
            else:
                time.sleep(0.1)

            #print(taskset_str + cmd.format(RPS=self.RPS, threads=self.n_threads))
            if subprocess.call(taskset_str + cmd.format(RPS=self.RPS, threads=self.n_threads), shell=True):
                raise Exception("{} launch failure!".format(self.name))
        # print_color("{} warmup".format(self.name), "green")
        time.sleep(WARMUP_TIME[self.name])
        self.pid = self.get_pid()
        self.mode = MODE.Unmanaged
        self.monitor()
        print_color("==> "+self.name + ' launched! pid:{}'.format(self.pid), "green")
        logger.info(self.name + ' launched! pid:{}'.format(self.pid))

    def change_RPS(self):
        if len(self.RPS_to_change) > 0:
            RPS = self.RPS_to_change.pop(0)
            cmd = CHANGE_RPS_STR[self.name].format(RPS=RPS["RPS"])
            if subprocess.call(cmd, shell=True):
                raise Exception("{} change RPS failure!".format(self.name))
            print_color("==> RPS of " + self.name + " is changed to {}.".format(RPS), "green")
            logger.info("RPS of " + self.name + " is changed to {}.".format(RPS))
            self.RPS_str = RPS["RPS_str"]

    def end(self):
        if self.pid is None:
            raise Exception("self.pid is None")
        if isinstance(self.pid, list):
            for pid in self.pid:
                subprocess.call("kill {}".format(pid), shell=True)
            else:
                subprocess.call("kill {}".format(self.pid), shell=True)
        self.mode = MODE.Dead

    def add_RPS_to_change(self, RPS, RPS_str, time):
        self.RPS_to_change.append({"RPS": RPS, "time": time, "RPS_str":RPS_str})
        self.RPS_to_change.sort(key=lambda x: x["time"])

    def get_pid(self):
        if self.name in ["nginx"]:
            self.pid = [int(item) for item in check_output(
                ["pgrep", "-f", NAME_2_PNAME[self.name]]).split()]
            self.pid_str = ",".join([str(pid) for pid in self.pid])
            self.pid_str_md5 = hashlib.md5(self.pid_str.encode()).hexdigest()
        else:
            self.pid = int(check_output(
                ["pgrep", "-n", "-f", NAME_2_PNAME[self.name]]))
            self.pid_str = str(self.pid)
            self.pid_str_md5 = hashlib.md5(self.pid_str.encode()).hexdigest()
        return self.pid

    def get_mode(self):
        return self.mode

    def get_latency(self):
        return self.get_features(["Latency"])[0]

    def get_QoS(self):
        return QOS_TARGET[self.name]/self.get_latency()

    def get_avg_history_latency(self):
        return np.mean(self.history_latency)

    def get_avg_history_IPS(self):
        return np.mean(self.history_IPS)

    def is_QoS_met(self):
        QoS_target = QOS_TARGET[self.name]
        return self.get_avg_history_latency() <= QoS_target or self.get_features(["Latency"])[0] <= QoS_target

    def is_latency_improved(self):
        return self.get_latency() < self.get_avg_history_latency()

    def get_latency_improvement(self):
        return self.get_latency() / self.get_avg_history_latency()

    def is_IPS_decreased(self):
        return self.get_features(["IPS"])[0] < self.get_avg_history_IPS()

    def is_under_provision(self):
        if AGGRESSIVE:
            return not self.is_QoS_met() and self.get_latency_improvement() > 0.95
        else:
            return not self.is_QoS_met() and not self.is_latency_improved()

    def is_over_provision(self):
        if AGGRESSIVE:
            return self.is_QoS_met() and self.get_QoS_violation() < -0.2
        else:
            return self.is_QoS_met() and self.is_latency_improved() and self.is_IPS_decreased()

    def is_cores_under_provision(self):
        if self.core_len is not None:
            return self.get_features("CPU_Utilization")[0] >= 0.95 * self.core_len * 100
        else:
            return False

    def get_QoS_violation(self):
        QoS_violation = self.get_avg_history_latency() / \
            QOS_TARGET[self.name] - 1
        return QoS_violation

    def monitor(self):
        tail = "" if PQOS_OUTPUT_ENABLED else " 1>/dev/null"
        self.pid = self.get_pid()
        if not os.path.exists(self.TMP_DIR):
            os.makedirs(self.TMP_DIR)
        pqosout = open(self.TMP_DIR + "pqos.{}".format(self.pid_str_md5), "w")
        topout = open(self.TMP_DIR + "top.{}".format(self.pid_str_md5), "w")
        if isinstance(self.pid, list):
            if self.name in ["nginx"]:
                subprocess.call('top -d {} -b -w 512 | grep -E "COMMAND|nobody.*{}" &'.format(
                    0.5, self.name), shell=True, stdout=topout)
            else:
                subprocess.call('top -d {} -b -w 512 | grep -E "COMMAND|{}" &'.format(
                    0.5, self.name), shell=True, stdout=topout)
        else:
            subprocess.call('top -p {} -d {} -b -w 512 &'.format(self.pid_str,
                            TOP_INTERVAL, self.name), shell=True, stdout=topout)
        subprocess.call("pqos -I -a pid:{}={}".format(self.COS_id,
                        self.pid_str) + tail, shell=True)
        subprocess.call(
            "pqos -I -i 1 -p all:{} &".format(self.pid_str), shell=True, stdout=pqosout)
        time.sleep(1)

    def set_state(self, keys):
        for key in keys:
            self.state[key] = None
        if any([key in ["Virt_Memory", "Res_Memory", "CPU_Utilization", "Memory_Footprint"] for key in keys]):
            if isinstance(self.pid, int):
                mem_res = check_output(
                    "tail -n 1 {}top.{}".format(self.TMP_DIR, self.pid_str_md5), shell=True).decode()
                mem_split = mem_res.split()
                self.state["Virt_Memory"] = any_2_byte(mem_split[4])
                self.state["Res_Memory"] = any_2_byte(mem_split[5])
                self.state["CPU_Utilization"] = float(mem_split[8])
                self.state["Memory_Footprint"] = float(mem_split[9])
            elif isinstance(self.pid, list):
                mem_res = check_output("tail -n {} {}top.{}".format(
                    (len(self.pid)+1)*2, self.TMP_DIR, self.pid_str), shell=True).decode()
                if len(mem_res) == 0:
                    return False
                mem_lines = [line.split() for line in mem_res.split("\n")]
                if len(mem_lines[-1]) < 12:
                    mem_lines = mem_lines[:-1]
                headline_indices = []
                for idx in range(len(mem_lines)):
                    if mem_lines[idx][-1] == "COMMAND":
                        headline_indices.append(idx)
                if (len(mem_lines)-headline_indices[-1]) < len(self.pid)+1:
                    mem_lines = mem_lines[headline_indices[-2]:headline_indices[-2]+len(self.pid)+1]
                else:
                    mem_lines = mem_lines[headline_indices[-1]:headline_indices[-1]+len(self.pid)+1]

                all_values = defaultdict(list)
                for line in mem_lines[1:]:
                    all_values["Virt_Memory"].append(any_2_byte(line[4]))
                    all_values["Res_Memory"].append(any_2_byte(line[5]))
                    all_values["CPU_Utilization"].append(float(line[8]))
                    all_values["Memory_Footprint"].append(float(line[9]))
                self.state["Virt_Memory"] = np.mean(all_values["Virt_Memory"])
                self.state["Res_Memory"] = np.mean(all_values["Res_Memory"])
                self.state["CPU_Utilization"] = sum(
                    all_values["CPU_Utilization"])
                self.state["Memory_Footprint"] = sum(
                    all_values["Memory_Footprint"])

        if any([key in ["IPC", "Misses", "LLC", "MBL"] for key in keys]):
            if self.name in ["nginx"]:
                pqos_res = check_output(
                    "tail -n {} {}pqos.all".format((N_CORES+2)*2, self.TMP_DIR), shell=True).decode().strip()
                pqos_lines = [line.split() for line in pqos_res.split("\n")]
                time_indices = []
                for idx in range(len(pqos_lines)):
                    if pqos_lines[idx][0] == "TIME":
                        time_indices.append(idx)
                if (len(pqos_lines)-time_indices[-1]) < N_CORES+2:
                    pqos_lines = pqos_lines[time_indices[-2]:time_indices[-2]+N_CORES+2]
                else:
                    pqos_lines = pqos_lines[time_indices[-1]:time_indices[-1]+N_CORES+2]

                all_values = defaultdict(list)
                for line_index, line in enumerate(pqos_lines[2:]):
                    if line_index >= self.core_start and line_index <= self.core_start + self.core_len - 1:
                        all_values["IPC"].append(float(line[1]))
                        all_values["Misses"].append(int(line[2].rstrip("k")))
                        all_values["LLC"].append(float(line[3]))
                        all_values["MBL"].append(float(line[4]))
                self.state["IPC"] = np.mean(all_values["IPC"])
                self.state["Misses"] = sum(all_values["Misses"])
                self.state["LLC"] = sum(all_values["LLC"])
                self.state["MBL"] = sum(all_values["MBL"])
            else:
                pqos_res = check_output(
                    "tail -n 3 {}pqos.{}".format(self.TMP_DIR, self.pid_str_md5), shell=True).decode()
                pqos_split = pqos_res.split()
                self.state["IPC"] = pqos_split[12]
                self.state["Misses"] = pqos_split[13].rstrip("k")
                self.state["LLC"] = pqos_split[14]
                self.state["MBL"] = pqos_split[15]

        if any([key in ["Allocated_Core", "Allocated_Cache"] for key in keys]):
            core_list = []
            way_list = []
            if not (self.core_start is None or self.core_len is None):
                core_list.extend(
                    list(range(self.core_start, self.core_start + self.core_len)))
            for proc_id_str, policy in self.sharing_policies.items():
                core_list.extend(list(range(policy["position"]["core_start"],
                                            policy["position"]["core_start"] + policy["position"]["core_len"])))
            if not (self.way_start is None or self.way_len is None):
                way_list.extend(
                    list(range(self.way_start, self.way_start + self.way_len)))
            for proc_id_str, policy in self.sharing_policies.items():
                way_list.extend(list(range(policy["position"]["way_start"],
                                           policy["position"]["way_start"] + policy["position"]["way_len"])))

            self.state["Allocated_Core"] = len(
                core_list) if len(core_list) > 0 else N_CORES
            self.state["Allocated_Cache"] = way_2_cache(len(way_list), MB_PER_WAY) if len(way_list) > 0 else way_2_cache(N_WAYS, MB_PER_WAY)
            if len(core_list) == 0:
                core_list = list(range(0, N_CORES))

        if any([key in ["Frequency"] for key in keys]):
            freq_sum = 0
            freq_res = check_output(
                "cat /proc/cpuinfo| grep 'cpu MHz'", shell=True)
            freq_line = freq_res.splitlines()
            for i in core_list:
                freq_temp = re.findall(r"\d+\.?\d*", freq_line[i].decode())
                freq_sum += float(freq_temp[0])
            self.state["Frequency"] = round(freq_sum / len(core_list), 1)

        if self.regular_update:
            Latency_update = self.last_latency is None or time.time(
            ) - self.last_latency[0] >= LATENCY_INTERVAL
        else:
            Latency_update = True

        if any([key in ["Latency"] for key in keys]):
            if Latency_update:
                try:
                    cmd = LATENCY_STR[self.name]
                    output = check_output(cmd, shell=True).decode()
                    if self.name in ["img-dnn", "xapian", "moses", "sphinx", "specjbb", "masstree", "login", "silo", "ads"]:
                        latency = float(output) / 1000000
                    elif self.name in ["mongodb", "mysql", "redis"]:
                        if "[]" in output:
                            latency = None
                        else:
                            latency = float(output.split(
                                ", ")[-3].split("=")[1]) / 1000
                    elif self.name in ["nginx", "nodejs"]:
                        latency = float(output) / 1000
                    elif self.name in ["memcached"]:
                        latency = float(output.split()[-1]) / 1000
                    else:
                        latency = None
                except Exception as e:
                    latency = None
                if latency is not None:
                    self.last_latency = (time.time(), latency)
                    self.history_latency.append(latency)
                    self.state["Latency"] = latency
            else:
                self.state["Latency"] = self.last_latency[1]

        if self.regular_update:
            IPS_update = self.last_IPS is None or time.time() - \
                self.last_IPS[0] >= PERF_INTERVAL
        else:
            IPS_update = True
        if any([key in ["IPS"] for key in keys]):
            if IPS_update:
                try:
                    IPS_res = check_output(
                        "perf stat -p {} -e instructions sleep 0.05 2>&1|grep instructions".format(self.pid), shell=True)
                    IPS = int("".join(IPS_res.decode().split()[0].split(",")))
                except ValueError as e:
                    IPS = None
                if IPS is not None:
                    self.last_IPS = (time.time(), IPS)
                    self.history_IPS.append(IPS)
                    self.state["IPS"] = IPS
            else:
                self.state["IPS"] = self.last_IPS[1]
        return True

    def get_features(self, keys):
        if not isinstance(keys, list):
            keys = [keys]
        success = self.set_state(keys)
        arr = np.array([self.state[key] for key in keys], dtype=float)
        return arr
