from cv2 import (
    imread, 
    cvtColor, 
    COLOR_BGR2RGB
)
from glob import glob
from numpy import (
    mean, 
    median,
    std
)
from os.path import basename
from pickle import load
from pandas import DataFrame


class Classificacao:
    
    def __init__(self, path_model_class='model_classific/RF_Model.pkl') -> None:
        self.clas_model = self.loadModel(path_model_class)
        

    def loadModel(self,name_model='Modelo.pkl'):
        """
        Carregar o modelo RandomForestClassifer salvo em um arquivo pkl e o retorna.
        
        Args:
            model_name::str: Nome do arquivo que será gerado
            
        Return:
            model::RF: Modelo RandomForestClassifer já treinado.
        """ 
        # 
        with open(name_model, 'rb') as arquivo:
            model = load(arquivo)
            return model
        
    def predict(self, path):
        
        novo_dado = self.extract_one(path=path)

        previsao = self.clas_model.predict(novo_dado)

        return bool(previsao)


    def extract_one(self,fname:str)->DataFrame:
        """
        Extrai as caracteristica de uma image que teve seu caminho passado como parâmetro e 
        arquiva os resultados em um DF

        Args:
            fname::str: Nome da imagem que será aberta e processada.

        Returns:
            df::DataFrame: DataFrame pandas contendo os resultados da extração.
        """

        lista_dados = []

        colunas = ['Media_Canal_R','Media_Canal_G','Media_Canal_B','Mediana_Canal_R','Mediana_Canal_G','Mediana_Canal_B','Desvio_Canal_R','Desvio_Canal_G','Desvio_Canal_B']


        imagem_BGR = imread(fname)

        imagem_RGB = cvtColor(imagem_BGR,COLOR_BGR2RGB)


        media_R = round(mean(imagem_RGB[:,:,0]))
        media_G = round(mean(imagem_RGB[:,:,1]))
        media_B = round(mean(imagem_RGB[:,:,2]))

        mediana_R = round(median(imagem_RGB[:,:,0]))
        mediana_G = round(median(imagem_RGB[:,:,1]))
        mediana_B = round(median(imagem_RGB[:,:,2]))

        desvio_R = round(std(imagem_RGB[:,:,0]))
        desvio_G = round(std(imagem_RGB[:,:,1]))
        desvio_B = round(std(imagem_RGB[:,:,2]))


        lista_dados.append([media_R,media_G,media_B,mediana_R,mediana_G,mediana_B,desvio_R,desvio_G,desvio_B])

        df = DataFrame(data=lista_dados,columns=colunas)

        return df


    
    def export(self,path_dir:str,mode:str)->None:
        """
        Processa uma página de imagens FAMACHA e salva o resultado no formato escolhido pelo usuário. 
        Opções válidas para mode [csv, excel, json]

        Args:
            path_dir::str: Caminho para a pasta contendo as imagens que serão processadas.
            mode::str: Palavra chave que define como serão salvos os dados processados.
            
        Return:
            None
        """
        
        match mode.lower():
            case 'csv':
                pass
            
            case 'excel':
                pass
            
            case 'json':
                pass
    
    