from pickle import load
from pandas import DataFrame
from numpy import argmax

def PKL_Model(name_model: str ='Modelo.pkl'):
    """
    Carregar o modelo do sklearn salvo em um arquivo .pkl apartir do seu caminho passado, e o retorna, pronto para ser usado em uma variável.
    """
    # 
    with open(name_model, 'rb') as arquivo:
        model = load(arquivo)
    return model

def PKL_classify(modelo: object, df:DataFrame)->list:
    """
    Usa um modelo sklearn passado no parâmetro para realizar a predição do grau Famacha em um conjunto de dados em formato DataFrame pandas
    
    Retorna:
        - Lista vazia em caso do dataFrame estar vazio
        - Lista, onde cada elemento corresponde a predição da situação FAMACHA (0,1) para cada registro no DataFrame
    """
    predicts = []
    
    for _, row in df.iterrows():
        data = row[['Mean_R', 'Mean_G', 'Mean_B', 'Median_R', 'Median_G', 'Median_B', 'Std_R', 'Std_G', 'Std_B']].values.reshape(1, -1)
        
        predict_proba = modelo.predict_proba(data) #Retorna a probabilidade de todas as classes
        
        predict_class = argmax(predict_proba, axis=1) # pegamos o indice da maior probabilidade
        predicts.append(predict_class[0]) # pegamos a classe com a maior probabilidade
        #print(f"Linha 30\nPredict Prova: {predict_proba}\nClasse: {predict_class}\nReturn: {predicts}\n")
    return predicts