from cv2 import bitwise_and,fillPoly, resize
from typing import List, Tuple
from ultralytics import YOLO
from numpy import array, int32, ndarray, uint8,zeros


def SegModel(filePath:str = r"models\best.pt") -> YOLO | None:
    """
    Recebe o caminho de um modelo YOLO e o retorna
    Caso falhe ao encontrar o modelo retorna None
    
    Args:
        filepath::str: Caminho para o arquivo .pt
        
    Return:
        Model:: Yolo | None: Retorna o modelo carrega ou None em caso de falha
        
    """
    try:
        model = YOLO(filePath)
    except:
        model = None
        
    return model

def ResizeList(images:List[ndarray], size:Tuple[int,int] = (512,512))-> List[ndarray]: #(512,512), (256,256), (128,128), (64,64)
    """
    Redimensiona uma lista de imagens para a proporção desejada
    
    Args:
        images::List[ndarray]: Lista de imagens no formato ndarray a serem redimensionadas
        size::Tuple[int,int]: Tupla contendo 2 valores usadaos para definir as novas dimensões, sendo eles respectivamente largura e altura
        
    Return:
        resized::List[ndarray]: Lista contendo as imagens redimensionadas
    """
    resized = []
    
    for imagem in images:
        resized.append(resize(imagem, size))
        
    return resized

def Predict_image(image,modelsPath:str="best.pt",conf:float=0.5):
        """
        Processa uma imagem e retorna um dicionário com os dados obtidos.
        O dicionário possui as seguintes chaves -> xyxys,confidences,class_id,masks,probs
        
        Args:
            image::str|ndarray: Nome de uma imagem processada para o recorte
            modelsPath::str: Caminho para o modelo YOLO salvado no formato .pt para segmentação
            confiance::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
            
        Return:
            dic::dict: Dicionário Contendos os dados obtidos no processamento
        """
        
        dic = dict()
        
        try:
            model = YOLO(modelsPath)
            results = model.predict(image,conf=conf,boxes=False,max_det=1)

            result = results[0]
            boxes = result.boxes.cpu().numpy()
        
            dic['xyxys'] = boxes.xyxy
            dic['confidences'] = boxes.conf
            #dic['masks'] = (result.masks.xy,result.masks.data)
            dic['masks'] = result.masks.xy

            
        except:
            dic = None
        
        return dic


def Segment(image:ndarray,model: YOLO)->ndarray | None:
    """
    Recebe uma imagem RGB famacha e retorna:
    - None, caso não seja uma imagem valida
    - ndarray, Segmentação da zona de interesse coletada.
    
    """
    try:
        results = model.predict(source=image,boxes=False,conf=0.3,max_det=1)

        result = results[0]
        xy = result.masks.xy

        mask = zeros(image.shape[:2], dtype=uint8)

        # Converter a lista de tuplas em um array numpy
        pts = array([tuple(map(int, ponto)) for array in xy for ponto in array], dtype=int32)

        # Desenhar a região de interesse na máscara
        fillPoly(mask, [pts], (255))  # Preenche a região da máscara com branco

        # Aplicar a máscara na imagem original
        segmentacao = bitwise_and(image, image, mask=mask)
        
    except:
        segmentacao = None
    
    return segmentacao



def SegmentedList(images:List[ndarray], model:YOLO, is_resized: bool = False, new_size: Tuple[float,float] = (512,512))->List[ndarray]:
    """
    Recebe uma lista de imagens e realizada a segmentação em cada uma das imagens.
    A função pode realizar o redimensionamento da imagem caso a mesma não esteja no padrão adequado (512,512)
    - Caso a imagem seja uma imagem válida, retorna a zona de interesse obtida pela segmentação.
    - Caso não seja, retorna um valor Nulo para aquela imagem
    
    """
    if is_resized == False:
        images = ResizeList(images,size=new_size)
    
    segmented = []
    
    for image in images:
        segmented.append(Segment(image,model))
        
    return segmented



