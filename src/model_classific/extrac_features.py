from numpy import mean, median,std
from cv2 import (
    imread, 
    cvtColor, 
    COLOR_BGR2RGB
)
from pandas import DataFrame
from os.path import basename

def extract_all(glob_dir:list)->DataFrame:
    """
    Extrai as caracteristica de uma base de dados passada como parâmetro e 
    arquiva os resultados em um DF
    
    Args:
        glob_dir::list: Lista contendo o caminho de cada imagem permitindo as mesmas serem abertas.
    
    Return:
        df::DataFrame: DataFrame pandas contendo os resultados da extração.
    """

    lista_dados = []

    colunas = ['Nome_Imagem','Vermifuga','Media_Canal_R','Media_Canal_G','Media_Canal_B','Mediana_Canal_R','Mediana_Canal_G','Mediana_Canal_B','Desvio_Canal_R','Desvio_Canal_G','Desvio_Canal_B']


    for file in glob_dir:

        imagem_BGR = imread(file)

        imagem_RGB = cvtColor(imagem_BGR,COLOR_BGR2RGB)


        media_R = round(mean(imagem_RGB[imagem_BGR[:,:,0]!= 0, 0]))
        media_G = round(mean(imagem_RGB[imagem_BGR[:,:,1]!= 0, 1]))
        media_B = round(mean(imagem_RGB[imagem_BGR[:,:,2]!= 0, 2]))

        #calculando a mediana nos canais de cor
        mediana_R = round(median(imagem_RGB[imagem_BGR[:,:,0]!= 0, 0]))
        mediana_G = round(median(imagem_RGB[imagem_BGR[:,:,1]!= 0, 1]))
        mediana_B = round(median(imagem_RGB[imagem_BGR[:,:,2]!= 0, 2]))

        #calculando o desvio padrão nos canais de cor

        desvio_R = round(std(imagem_RGB[imagem_BGR[:,:,0]!= 0, 0]))
        desvio_G = round(std(imagem_RGB[imagem_BGR[:,:,1]!= 0, 1]))
        desvio_B = round(std(imagem_RGB[imagem_BGR[:,:,2]!= 0, 2]))

        #preenchendo os dados da imagem em uma lista  de listas

        vermifuga_S_N = int(file[-5])

        if vermifuga_S_N > 0 and vermifuga_S_N <3:
            resultado = False
        elif vermifuga_S_N > 2 and vermifuga_S_N <6:
            resultado = True
        else:
            resultado = 'Falha'

        lista_dados.append([basename(file),resultado,media_R,media_G,media_B,mediana_R,mediana_G,mediana_B,desvio_R,desvio_G,desvio_B])


    df = DataFrame(data=lista_dados,columns=colunas)

    return df


def extract_one(fname:str)->DataFrame:
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
