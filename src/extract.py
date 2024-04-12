from numpy import ndarray, mean, median, std
from typing import List
from pandas import DataFrame

def Images2DF(images:List[ndarray])->DataFrame:
    """
    Extrai as caracteristica de uma lista de imagens contendo uma ou mais imagens 
    e arquiva os resultados em um DataFrame Pandas

    Args:
        images::List[ndarray]: Lista de imagens que serão processadas

    Returns:
        df::DataFrame: DataFrame pandas contendo os resultados da extração.
    """
    #print("linha 16, images:\n",images)
    lista_dados = []

    colunas = ['Mean_R','Mean_G','Mean_B','Median_R','Median_G','Median_B','Std_R','Std_G','Std_B']


    for image_RGB in images:
        #print("linha 23, image_rgb:\n", image_RGB.shape)
        # canais de cor com valor zero serão ignorados, pois a segmentação garante que a zona de interesse não tenha valor 0
        
        # Calculando a médianos canais de cor 
        mean_R = round(mean(image_RGB[image_RGB[:,:,0]!= 0, 0]),2)
        mean_G = round(mean(image_RGB[image_RGB[:,:,1]!= 0, 1]),2)
        mean_B = round(mean(image_RGB[image_RGB[:,:,2]!= 0, 2]),2)

        #calculando a mediana nos canais de cor
        median_R = round(median(image_RGB[image_RGB[:,:,0]!= 0, 0]),2)
        median_G = round(median(image_RGB[image_RGB[:,:,1]!= 0, 1]),2)
        median_B = round(median(image_RGB[image_RGB[:,:,2]!= 0, 2]),2)

        #calculando o desvio padrão nos canais de cor
        std_R = round(std(image_RGB[image_RGB[:,:,0]!= 0, 0]),2)
        std_G = round(std(image_RGB[image_RGB[:,:,1]!= 0, 1]),2)
        std_B = round(std(image_RGB[image_RGB[:,:,2]!= 0, 2]),2)

        lista_dados.append([mean_R,mean_G,mean_B,median_R,median_G,median_B,std_R,std_G,std_B])

    df = DataFrame(data=lista_dados,columns=colunas)

    return df