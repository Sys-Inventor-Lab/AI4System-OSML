import pandas as pd
import sys
from tqdm import tqdm

sys.path.append("../../")
from utils import *
from configs import *


def normalization_and_split(path,dataset,features,labels):
    try:
        df=load_pkl(path+".pkl")
    except:
        df=pd.read_csv(path+".csv")
    if df is None:
        return
    print(df)
    df=df[features+labels]
    df=df.dropna(axis=0,how='any')
    for _,feature in enumerate(features):
        df[feature] = (df[feature] - MIN_VAL[feature]) / (MAX_VAL[feature] - MIN_VAL[feature])
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    #dump_pkl(path+"_normalized.csv",df)

    # 70%: train, 30%: test
    train_df = df.loc[:int(0.7 * df.shape[0])]
    test_df = df.loc[int(0.7 * df.shape[0]):]
    train_df.to_csv(path+"_train.csv".format(dataset),index=None)
    test_df.to_csv(path+"_test.csv".format(dataset),index=None)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ["Model_A", "Model_B"]
    func_map = {"Model_A":normalization_and_split, "Model_B":normalization_and_split}
    feature_map = {"Model_A":A_FEATURES,"Model_B":B_FEATURES}
    label_map = {"Model_A":A_LABELS,"Model_B":B_LABELS}
    for dataset in datasets:
        path = ROOT+"/data/data_process/{}/{}".format(dataset, dataset)
        func_map[dataset](path,dataset,feature_map[dataset],label_map[dataset])
