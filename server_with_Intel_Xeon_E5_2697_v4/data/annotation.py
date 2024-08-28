import pandas as pd
import sys
from tqdm import tqdm

sys.path.append("../")
sys.path.append("../../")
from utils import *

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)

N_CORES=36
CORE_INDEX = list(range(0, N_CORES))
N_WAYS=20
WAY_INDEX = list(range(0, N_WAYS))
MB_PER_WAY=2.25

def annotation(raw_path, label_root):
    name, n_thread, RPS = raw_path.split('/')[2:]
    label_path = label_root + "{}/{}/{}/".format(name, n_thread, RPS)
    if os.path.exists(label_path + 'fail.txt'):
        # Can not find labels for the collected traces.
        return None
    if os.path.exists(label_path + 'labels.txt'):
        # The labels exist, read from files directly.
        with open(label_path + 'labels.txt', 'r') as f:
            labels = eval(f.readline())

    else:
        # The labels do not exist.
        lat_df = pd.DataFrame(columns=CORE_INDEX, index=WAY_INDEX)
        dfs = {}
        empty = True
        for sub_path in walk(raw_path):
            name, n_thread, RPS, core_idx, way_idx = sub_path.split('/')[-1].split('+')
            core_idx, way_idx = int(core_idx), int(way_idx)
            try:
                df = pd.read_csv(sub_path)
            except:
                print(sub_path)
            df.sort_values(by="Latency")
            #latency = df.iloc[int(df.shape[0]*0.99)]["Latency"]
            latency=df["Latency"].mean()
            lat_df.loc[way_idx, core_idx] = latency
            dfs[(way_idx, core_idx)] = df
            empty = False
        if empty:
            return None
        RCliff = [CORE_INDEX[-1], WAY_INDEX[-1]]

        for core_idx in reversed(CORE_INDEX):
            for cache_idx in reversed(WAY_INDEX):
                if lat_df.loc[cache_idx, core_idx] < QOS_TARGET[name] and sum([core_idx, cache_idx]) < sum(RCliff):
                    RCliff = [core_idx, cache_idx]

        OAA = [RCliff[0] + 2, RCliff[1] + 2]

        print(lat_df.iloc[:, :N_CORES//2])
        print(lat_df.iloc[:, N_CORES//2:])

        print("name:", name)
        print("n_thread:", n_thread)
        print("RPS:", RPS)
        inp_str = input(
            'RCliff:\ncore_idx:{}; way_idx:{};\nOAA:\ncore_idx:{}; way_idx:{};\nOK? y(es)/n(o)/d(rop)/c(ancel), input your choice:'.format(
                RCliff[0], RCliff[1],
                OAA[0], OAA[1]))
        if (inp_str == 'n' or inp_str == 'N'):
            RCliff[0] = int(input('Index of RCliff cores:'))
            RCliff[1] = int(input('Index of RCliff cache ways:'))
            # OAA[0] = int(input('Index of OAA cores:'))
            # OAA[1] = int(input('Index of OAA cache ways:'))
            OAA[0] = RCliff[0] + 2
            OAA[1] = RCliff[1] + 2
        elif (inp_str == 'd'):
            os.system('mkdir -p {}'.format(label_path))
            with open(label_path + 'fail.txt', 'w') as f:
                f.write(str(False))
            return None
        elif (inp_str == "c"):
            return None
        labels = {A_LABELS[0]: way_2_cache(RCliff[1] + 1, MB_PER_WAY),
                  A_LABELS[1]: RCliff[0] + 1,
                  A_LABELS[2]: way_2_cache(OAA[1] + 1, MB_PER_WAY),
                  A_LABELS[3]: OAA[0] + 1}
        os.system('mkdir -p {}'.format(label_path))
        with open(label_path + 'labels.txt', 'w') as f:
            f.write(str(labels))


def get_labels(raw_path, label_root):
    name, n_thread, RPS = raw_path.split('/')[-3:]
    label_path = label_root + "{}/{}/{}/".format(name, n_thread, RPS)
    if not os.path.exists(label_path):
        return None
    if os.path.exists(label_path + 'fail.txt'):
        # Can not find labels for the collected traces.
        return None
    if os.path.exists(label_path + 'labels.txt'):
        # The labels exist, read from files directly.
        with open(label_path + 'labels.txt', 'r') as f:
            labels = eval(f.readline())
        return labels


if __name__ == '__main__':
    for root in ["single", "multiple"]:
        raw_root = "data_collection/" + root + "/"
        label_root = "data_process/labels_" + root + "/"
        paths=[]
        for path_1 in walk(raw_root):
            for path_2 in walk(path_1):
                for path_3 in walk(path_2):
                    paths.append(path_3)
        for path in tqdm(paths):
            annotation(path, label_root)
