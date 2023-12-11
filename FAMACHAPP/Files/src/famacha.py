from clas_model import Classificacao
from seg_model import Segmentacao

class Famacha:

    
    def __init__(self,path_model_class='model_classific/RF_Model.pkl',path_model_seg='model_segment/weights/best.pt'):
        
        self.classificacao = Classificacao(path_model_class)
        self.segmentacao = Segmentacao(path_model_seg)
    