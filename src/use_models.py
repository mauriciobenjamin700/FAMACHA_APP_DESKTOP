from pickle import load
from pandas import DataFrame
from numpy import array

def PKL_Model(name_model: str ='Modelo.pkl'):
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

def PKL_classify(modelo, df:DataFrame):
    
    predicts = []
    
    for _,row in df.iterrows():
        
        #print(array(list(row)[:]))
        #data = xgb.DMatrix(array([list(row)[1:]]))
        data = array(list(row)[:]).reshape(1,-1)#pegamos apartir do 1 para ignorar o nome da imagem
        #print(data)
        
        predicts.append(modelo.predict(data)[0]) #por algum motivo ele retorna um vetor ao inves de apenas um resultado para a predição. Provavelmente é um vetor de possibilidade para cada classe 

    return predicts