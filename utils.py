import os
import time
import warnings
import pickle as pkl
import numpy as np
import pandas as pd
from functools import wraps

def any_2_byte(val):
    if chr(val[-1] == "g"):
        res = float(val[:-1]) * 1024 * 1024
    elif chr(val[-1] == "m"):
        res = float(val[:-1]) * 1024
    else:
        res = float(val)
    return res


def cache_2_way(cache, mb_per_way):
    return int(round(cache / mb_per_way))


def way_2_cache(way, mb_per_way):
    return way * mb_per_way


def walk(path):
    path = path.rstrip("/")
    if not os.path.exists(path):
        return []
    sub_paths = []
    for dir in os.listdir(path):
        sub_paths.append(path + '/' + dir)
    return sub_paths


def print_color(text, color="red"):
    escape_code = {"red": "\033[31m{}\033[0m",
                   "green": "\033[32m{}\033[0m",
                   "yellow": "\033[33m{}\033[0m",
                   "blue": "\033[34m{}\033[0m",
                   "magenta": "\033[35m{}\033[0m",
                   "cyan": "\033[36m{}\033[0m"}
    if color in escape_code:
        print(escape_code[color].format(text))
    else:
        raise Exception(
            "Only red/green/yellow/blue/magenta/cyan are available.")


def timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[finished {func_name} in {time:.2f}s]'.format(
            func_name=function.__name__, time=t1 - t0))
        return result
    return function_timer


def load_pkl(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            item = pkl.load(f, protocol=pkl.HIGHEST_PROTOCOL)
            return item
    else:
        raise Exception("Path {} does not exist.".format(path))


def dump_pkl(path, item):
    with open(path, "wb") as f:
        pkl.dump(item, f, protocol=pkl.HIGHEST_PROTOCOL)



def draw_bar_chart(N, ranges, labels):
    total_units = sum(ranges)
    assert(total_units==N)

    bar_length = 100
    unit_length = bar_length / N

    bars = []
    for i in range(len(ranges)):
        units = ranges[i]
        bar_units = int(unit_length * units)
        bar = "=" * bar_units
        bars.append(bar)

    bar_line = "|"+"|".join(bars)+"|"
    print(bar_line)

    label_line = "|"
    prop_line = "|"
    for i in range(len(labels)):
        label = labels[i]
        label_line += label[:len(bars[i])].ljust(len(bars[i]))+"|"
        prop_line += "({}/{})".format(ranges[i],N)[:len(bars[i])].ljust(len(bars[i]))+"|"
    print(label_line)
    print(prop_line)
    print(bar_line)


class stored_data:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.load()

    def empty(self):
        return self.data is None

    def store(self):
        with open(self.path, "wb") as f:
            pkl.dump(self.data, f)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.data = pkl.load(f)
        else:
            warnings.warn("{} does not exists.".format(self.path))


class pct_map(stored_data):
    def __init__(self, path, load=True, init_values=0):
        stored_data.__init__(self, path)
        self.data = {}
        self.init_values = init_values
        if load:
            self.load()

    def check_map_exists(self, key_workload, key_scheduler):
        exists = True
        if not key_workload in self.data:
            self.data[key_workload] = {}
            exists = False
        if not key_scheduler in self.data[key_workload]:
            self.data[key_workload][key_scheduler] = np.zeros(
                (10, 10), dtype=np.int)
            if not self.init_values == 0:
                for i in range(10):
                    for j in range(10):
                        self.data[key_workload][key_scheduler][i][j] = self.init_values
            exists = False
        return exists

    def show_map(self, key_workload, key_scheduler):
        self.check_map_exists(key_workload, key_scheduler)
        print(self.data[key_workload][key_scheduler])

    def update_map(self, key_workload, key_scheduler, pcts, update_lt=False, update_gt=False):
        self.check_map_exists(key_workload, key_scheduler)
        if not update_lt and not update_gt:
            if self.data[key_workload][key_scheduler][int(pcts[0]/10)-1][int(pcts[1]/10)-1] < pcts[2]:
                self.data[key_workload][key_scheduler][int(
                    pcts[0]/10)-1, int(pcts[1]/10)-1] = pcts[2]
        if update_lt:
            for i in range(int(pcts[0] / 10)):
                for j in range(int(pcts[1] / 10)):
                    if self.data[key_workload][key_scheduler][i][j] < pcts[2]:
                        self.data[key_workload][key_scheduler][i][j] = pcts[2]
        if update_gt:
            for i in range(int(pcts[0]/10)-1, 10):
                for j in range(int(pcts[1]/10)-1, 10):
                    if self.data[key_workload][key_scheduler][i][j] > pcts[2]:
                        self.data[key_workload][key_scheduler][i][j] = pcts[2]

    def count_cases(self, key_workload, key_scheduler):
        self.check_map_exists(key_workload, key_scheduler)
        cases = 0
        for i in range(10):
            for j in range(10):
                cases += self.data[key_workload][key_scheduler][i][j]//10
        print("cases")
        print(cases)
        return cases

    def check_in(self, key_workload, key_scheduler, pcts):
        self.check_map_exists(key_workload, key_scheduler)
        if self.data[key_workload][key_scheduler][int(pcts[0] / 10 - 1)][int(pcts[1] / 10 - 1)] < pcts[2]:
            return False
        else:
            return True

    def check_out(self, key_workload, key_scheduler, pcts):
        self.check_map_exists(key_workload, key_scheduler)
        if self.data[key_workload][key_scheduler][int(pcts[0] / 10 - 1)][int(pcts[1] / 10 - 1)] < pcts[2]:
            return True
        else:
            return False

    def dump_csv(self):
        for key_workload in self.data:
            for key_scheduler in self.data[key_workload]:
                df = pd.DataFrame(data=self.data[key_workload][key_scheduler], columns=[
                                  (i+1)*10 for i in range(10)], index=[(i+1)*10 for i in range(10)])
                df.to_csv("{}+{}.csv".format(key_workload, key_scheduler))


