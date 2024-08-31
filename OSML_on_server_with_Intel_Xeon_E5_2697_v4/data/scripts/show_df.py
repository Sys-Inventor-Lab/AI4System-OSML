import sys
sys.path.append("../../")
from utils import *

def show_df(paths):
    if isinstance(paths,str):
        paths=[paths]
    for path in paths:
        df=load_pkl(path)
        print(df)

if __name__=="__main__":
    if len(sys.argv) > 1:
        show_df(sys.argv[1:])
