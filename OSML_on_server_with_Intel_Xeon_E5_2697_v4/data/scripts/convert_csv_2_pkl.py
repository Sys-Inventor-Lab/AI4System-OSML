import pandas as pd
import pickle
import os
import sys
sys.path.append("../../")
from configs import *

def convert_csv_2_pkl(path):
    pkl_path=path.replace(".csv",".pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path,"rb") as f:
            df=pickle.load(f)
    else:
        df=pd.read_csv(path,index_col=0)
    
    df=df.dropna(axis=0,how='any') 

    print(df.shape[0])

    with open(pkl_path,"wb") as f:
        pickle.dump(df,f)    

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path=sys.argv[1]
        convert_csv_2_pkl(path)

