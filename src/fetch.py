from cv2 import imread,cvtColor,COLOR_BGR2RGB
from numpy import ndarray
from typing import List, Tuple
from os.path import exists,join,basename
from glob import glob


def Image(filename:str="image.jpg")->ndarray:
    """
    Retorna uma imagem RGB em formato ndarray com base no item passado
    
    Args:
        filename::str: Caminho para a imagem que será carregada
        
    Return:
        file::ndarray: Imagem RGB carregada
    """
    
    file = imread(filename)
    
    if file is not None:
        if file.shape[2] == 4: #se for RGBA vamos dercartar o canal A para economizar memória
            file = file[:,:,:3]
        file = cvtColor(file,COLOR_BGR2RGB)
        
    return file


def Folder(foldername:str="images")->Tuple[List[ndarray],List[str]]:
    """
    Retorna uma lista de imagens ou uma lista vazia com base na pasta passada
    Caso consiga formar uma lista de imagens, retorna o rotulo de cada imagem 
    
    Args:
        foldername::str: Caminho para a pasta que será acessada
        
    Return:
        dataset::Tuple[List[ndarray],List[str]]: Lista de imagens e rotulos validas lida ou lista vazia caso não consiga ler
    """
    dataset = None
    
    if exists(foldername):
        
        images = []
        labels = []
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        files = []
            
        for e in extensions:
            files.extend(glob(join(foldername,e)))
            files.extend(glob(join(foldername,e.upper())))

        for file in files:
            images.append(Image(file))
            labels.append(basename(file))
            
        dataset = (images,labels)
        
    return dataset


