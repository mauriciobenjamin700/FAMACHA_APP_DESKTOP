from os.path import dirname, abspath, join
import sys

sys.path.append(dirname(abspath(__file__)))


from fetch import *
from crop import *
from extract import *
from use_models import *
from export import save_results


seg_model = join("src","models","YOLO.pt")
pre_model = join("src","models","RF.pkl")

#if __name__ == "__main__":
    #print("->",dirname(abspath(__file__)))
    #print(SegModel(seg_model))