from ultralytics import YOLO
from os.path import (
    basename,
    exists,
    join
)
from os import makedirs
from glob import glob
from cv2 import (
    bitwise_and,
    imwrite,
    imread, 
    INTER_AREA,
    fillPoly,
    resize,
)

import numpy as np

class Segmentacao:
    
    def __init__(self, path_model_seg='model_segment/weights/best.pt') -> None:
        self.seg_model = YOLO(path_model_seg)
        
    def predict_dir_image(self, list_fname,conf=0.5)->dict:
        """
        Processa uma imagem e retorna um dicionário de dicionários com os dados obtidos.
        Cada chave do dicionário é o nome de uma imagem, dicionário possui as seguintes chaves -> xyxys,confidences,class_id,masks,probs
        
        Parâmetros:
            fname::str: Nome de uma imagem processada para o recorte
            confiance::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
            
        Retorno:
            dic::dict: Dicionário Contendos os dados obtidos no processamento
        """
        
        json = {}
        
        try:
            results = self.seg_model.predict(list_fname,conf=conf,boxes=False,max_det=2)
             
            for idx,result in enumerate(results):
                
                dic = dict()
                boxes = result.boxes.cpu().numpy()

                dic['masks'] = result.masks
                dic['probs'] = result.probs
                dic['xyxys'] = boxes.xyxy
                dic['confidences'] = boxes.conf
                dic['class_id'] = boxes.cls
                
                json[basename(list_fname[idx])] = dic
        except:
            json = None
               
        
        return json
            

    def predict_image(self, fname:str,conf:float=0.5):
        """
        Processa uma imagem e retorna um dicionário com os dados obtidos.
        O dicionário possui as seguintes chaves -> xyxys,confidences,class_id,masks,probs
        
        Parâmetros:
            fname::str: Nome de uma imagem processada para o recorte
            confiance::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
            
        Retorno:
            dic::dict: Dicionário Contendos os dados obtidos no processamento
        """
        
        dic = dict()
        
        try:
            results = self.seg_model.predict(fname,conf=conf,boxes=False,max_det=2)

            result = results[0]
            boxes = result.boxes.cpu().numpy()
        
            dic['xyxys'] = boxes.xyxy
            dic['confidences'] = boxes.conf
            dic['class_id'] = boxes.cls
            #dic['masks'] = (result.masks.xy,result.masks.data)
            dic['masks'] = result.masks.xy
            
        except:
            dic = None
        
        return dic
            
            
    
    def axis_image(self,fname,confiance=0.5)->list:
        """
        Processa uma imagem e retorna os eixos x1,y1,x2,y2 que compõe os boxs que contem a zona de interesse da imagem.
        
        Parâmetros:
            fname::str: Nome de uma imagem processada para o recorte
            confiance::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
        
        Retorno:
            xyxys::list: Lista contendo tuplas com os eixos da imagem que estão nossa zona de interesse ou
            lista vazia caso não encontre nada
    
        """
        xyxys = []
        try:
            result = self.seg_model.predict(fname,conf=confiance,boxes=False,max_det=2,show_conf=False,show_labels=False)
            
            boxes = result[0].boxes.cpu().numpy()
            
            for xyxy in boxes.xyxy:
            
                xyxys.append((int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])))
        
        except:
            xyxys = None    
        return xyxys
    
    #recorta a imagem
    def snip_img(self,fname:str, confiance:float=0.5):
        """
        Processa uma imagem e retorna os pixels recortados da imagem.
        Onde os pixels compõe zonas de interesse que a imagem pode vir a possuir
        
        Parâmetros:
            fname::str: Nome de uma imagem processada para o recorte
            confiance::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
        
        Retorno:
            interest_region::list: Lista contendo as partes da imagem que estão nossa zona de interesse ou
            lista vazia caso não encontre nada
    
        """
        interest_region = []
        try:
            
            image = imread(fname)
            
            xyxys = self.axis_image(fname=fname,confiance=confiance)
            if len(xyxys) > 0:
                for xyxy in xyxys:
                    x1,y1,x2,y2 = xyxy
                    interest_region.append(image[y1:y2, x1:x2])
        except:
            interest_region = None
            
        return interest_region
    
    
    def resize(self,fname,width=640,height=640):
        img = resize(imread(fname),(width,height),interpolation=INTER_AREA)
        return img
    
    
    def segment_img(self,fname:str):
        """
        Recebe uma imagem famacha, a segmenta e retorna a zona de interesse coletada após a segmentação.
        
        Parâmetros:
            fname::str: Nome do arquivo que será segmentado
            
        Retorno:
            segmentacao:: numpy array contendo a imagem segmentada a ser retornada ou Nada caso não haja oq segementar na imagem
        """
        
        segmentacao = None
        
        try:
            dados = self.predict_image(fname=fname)
         
            xy = dados["masks"]

            img = imread(fname)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)

            # Converter a lista de tuplas em um array numpy
            pts = np.array([tuple(map(int, ponto)) for array in xy for ponto in array], dtype=np.int32)

            # Desenhar a região de interesse na máscara
            fillPoly(mask, [pts], (255))  # Preenche a região da máscara com branco

            # Aplicar a máscara na imagem original
            segmentacao = bitwise_and(img, img, mask=mask)
        
        except:
            pass
        
        return segmentacao
    
    def segment_dir_image(self,dir_images,dir_output):
        fnames = glob(join(dir_images,'*.jpg'))
        fail = join(dir_output,'fail')
        if not exists(fail):
            makedirs(fail)
        
        for file in fnames:
            image = self.segment_img(file)
            try:
                imwrite(join(dir_output,basename(file)), image)
            except:
                imwrite(join(fail,basename(file)), imread(file))
                