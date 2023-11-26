from ultralytics import YOLO
from os.path import basename,join,exists
from os import makedirs
from shutil import rmtree
from glob import glob
import cv2
import numpy as np

class Famacha:

    
    def __init__(self, path_model='model_segment/weights/best.pt') -> None:
        self.model = YOLO(path_model)
        
    def predict_dir_image(self, list_fname,conf=0.5)->dict:
        """
        Processa uma imagem e retorna um dicionário de dicionários com os dados obtidos.
        Cada chave do dicionário é o nome de uma imagemO dicionário possui as seguintes chaves -> xyxys,confidences,class_id,masks,probs
        
        Parâmetros:
            fname::str: Nome de uma imagem processada para o recorte
            confiance::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
            
        Retorno:
            dic::dict: Dicionário Contendos os dados obtidos no processamento
        """
        
        json = {}
        
        try:
            results = self.model.predict(list_fname,conf=conf,boxes=False,max_det=2)
             
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
            results = self.model.predict(fname,conf=conf,boxes=False,max_det=2)

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
            
            
    def mark_image(self,fname, confiance=0.5):
        if exists('runs'):
            rmtree('runs')
        results = self.model.predict(fname,save=True,conf=confiance,max_det=2,show_conf=True,show_labels=True)
        del results
    
    def mark_dir_image(self,path:str, conf:float=0.5)->None:
        """
        Acessa um diretório de imagens e pega todas as imagens com a extenção .jpg
        Salva na pasta 'runs/segment/predict' os resultados da marcação'
        Salva uma pasta runs/segment/predict/labels'
        
        Parâmetros:
            path::str: Diretório da pasta onde as imagens estão
            conf::float: Valor representando percetual váriando entre 0 e 1
            
        Retorno:
            Função não retorna nada
        """
        data = glob(join(path,'*.jpg'))
        if exists('runs'):
            rmtree('runs')
        for image in data:
            results = self.model.predict(image,save=True,conf=conf,imgsz=(640,640),max_det=2,show_conf=True,show_labels=True,save_txt=True,save_conf=True)
            print(f"\nTerminei {image}")
        del results, data
        
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
            result = self.model.predict(fname,conf=confiance,boxes=False,max_det=2,show_conf=False,show_labels=False)
            
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
            
            image = cv2.imread(fname)
            
            xyxys = self.axis_image(fname=fname,confiance=confiance)
            if len(xyxys) > 0:
                for xyxy in xyxys:
                    x1,y1,x2,y2 = xyxy
                    interest_region.append(image[y1:y2, x1:x2])
        except:
            interest_region = None
            
        return interest_region
    
    
    def resize(self,fname,width=640,height=640):
        img = cv2.resize(cv2.imread(fname),(width,height),interpolation=cv2.INTER_AREA)
        return img
    
    def rotate(self,fname)->tuple:
        
        img = cv2.imread(fname)

        (h, w) = img.shape[:2]

        center = (w / 2, h / 2)
        
        angle90 = 90
        angle180 = 180
        angle270 = 270
        
        scale = 1.0
        
        M = cv2.getRotationMatrix2D(center, angle90, scale)
        rotated90 = cv2.warpAffine(img, M, (h, w))
        
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated180 = cv2.warpAffine(img, M, (w, h))
        
        M = cv2.getRotationMatrix2D(center, angle270, scale)
        rotated270 = cv2.warpAffine(img, M, (h, w))
        
        return (rotated90,rotated180,rotated270)
    
    
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

            img = cv2.imread(fname)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)

            # Converter a lista de tuplas em um array numpy
            pts = np.array([tuple(map(int, ponto)) for array in xy for ponto in array], dtype=np.int32)

            # Desenhar a região de interesse na máscara
            cv2.fillPoly(mask, [pts], (255))  # Preenche a região da máscara com branco

            # Aplicar a máscara na imagem original
            segmentacao = cv2.bitwise_and(img, img, mask=mask)
        
        except:
            pass
        
        return segmentacao
    
    def segment_dir_image(self,dir_images,dir_output):
        fnames = glob(dir_images)
        
        for file in fnames:
            image = self.segment_img(file)
            if type(image) != None:
                cv2.imwrite(join(dir_output,basename(file)), image)
            else:
                falha = join(dir_output,'falhas')
                if not exists(falha):
                    makedirs(falha)
                    
                cv2.imwrite(join(falha,basename(file)), cv2.imread(file))
                