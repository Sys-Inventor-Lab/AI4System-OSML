import pandas as pd
import sys
from tqdm import tqdm

sys.path.append("../")
from utils import *
from configs import *


def normalization_and_split(path,features,labels):
    df=pd.read_csv(path+".csv")
    if df is None:
        return
    print(df)
    df=df[features+labels]
    df=df.dropna(axis=0,how='any')
    max_min_path=ROOT+"/data/data_process/max_min.txt"
    if os.path.exists(max_min_path):
        with open(max_min_path,"r") as f:
            max_min=eval(f.readline())
    else: 
        raise Exception("Please run python get_max_min.py to get the max_min.txt")

    for _,feature in enumerate(features):
        df[feature] = (df[feature] - max_min["min"][feature]) / (max_min["max"][feature] - max_min["min"][feature])
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)

    # 70%: train, 30%: test
    train_df = df.loc[:int(0.7 * df.shape[0])]
    test_df = df.loc[int(0.7 * df.shape[0]):]
    train_df.to_csv(path+"_train.csv",index=None)
    test_df.to_csv(path+"_test.csv",index=None)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ["Model_A", "Model_A_shadow", "Model_B", "Model_B_shadow"]
    func_map = {"Model_A":normalization_and_split, "Model_A_shadow":normalization_and_split, "Model_B":normalization_and_split, "Model_B_shadow":normalization_and_split}
    feature_map = {"Model_A":A_FEATURES, "Model_A_shadow":A_SHADOW_FEATURES, "Model_B":B_FEATURES, "Model_B_shadow":B_SHADOW_FEATURES}
    label_map = {"Model_A":A_LABELS, "Model_A_shadow":A_LABELS,"Model_B":B_LABELS,"Model_B_shadow":B_SHADOW_LABELS}
    for dataset in datasets:
        path = "data_process/{}/{}".format(dataset, dataset)
        func_map[dataset](path,feature_map[dataset],label_map[dataset])
