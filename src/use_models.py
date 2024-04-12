from pickle import load
from pandas import DataFrame
from numpy import argmax

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

def PKL_classify(modelo, df:DataFrame)->list:
    predicts = []
    
    for _, row in df.iterrows():
        data = row[['Mean_R', 'Mean_G', 'Mean_B', 'Median_R', 'Median_G', 'Median_B', 'Std_R', 'Std_G', 'Std_B']].values.reshape(1, -1)
        
        predict_proba = modelo.predict_proba(data) #Retorna a probabilidade de todas as classes
        
        predict_class = argmax(predict_proba, axis=1) # pegamos o indice da maior probabilidade
        predicts.append(predict_class[0]) # pegamos a classe com a maior probabilidade
        #print(f"Linha 30\nPredict Prova: {predict_proba}\nClasse: {predict_class}\nReturn: {predicts}\n")
    return predicts