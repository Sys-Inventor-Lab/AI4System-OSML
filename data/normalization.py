import pandas as pd
import sys
from tqdm import tqdm

sys.path.append("../")
from utils import *


def normalization_and_split(path,features,labels,update_max_min=False):
    try:
        df=load_pkl(path+".pkl")
    except:
        df=pd.read_csv(path+".csv")
    if df is None:
        return
    print(df)
    df=df[features+labels]
    df=df.dropna(axis=0,how='any')
    max_min_path="data_process/max_min/max_min_{}.txt".format(dataset)
    if os.path.exists(max_min_path) and update_max_min==False:
        with open(max_min_path,"r") as f:
            max_min=eval(f.readline())
    else: 
        df_max, df_min = df[features].max(), df[features].min()
        max_min = {"max": df_max.to_list(), "min": df_min.to_list()}
        with open(max_min_path, 'w') as f:
             f.write(str(max_min))

    for _,feature in enumerate(features):
        df[feature] = (df[feature] - max_min["min"][_]) / (max_min["max"][_] - max_min["min"][_])
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    #dump_pkl(path+"_normalized.csv",df)

    # 70%: train, 30%: test
    train_df = df.loc[:int(0.7 * df.shape[0])]
    test_df = df.loc[int(0.7 * df.shape[0]):]
    dump_pkl(path + '_train.pkl',train_df)
    dump_pkl(path + "_test.pkl",test_df)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    else:
        datasets = ["Model_A", "Model_A_shadow", "Model_B", "Model_B_shadow", "Model_C"]
    func_map = {"Model_A":normalization_and_split, "Model_A_shadow":normalization_and_split, "Model_B":normalization_and_split, "Model_B_shadow":normalization_and_split}
    feature_map = {"Model_A":A_FEATURES, "Model_A_shadow":A_SHADOW_FEATURES, "Model_B":B_FEATURES, "Model_B_shadow":B_SHADOW_FEATURES}
    label_map = {"Model_A":A_LABELS, "Model_A_shadow":A_LABELS,"Model_B":B_LABELS,"Model_B_shadow":B_SHADOW_LABELS}
    for dataset in datasets:
        path = "data_process/{}/{}".format(dataset, dataset)
        func_map[dataset](path,feature_map[dataset],label_map[dataset])
