from extrac_features import extract_one
from pandas import read_excel
from pickle import dump,load
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    cohen_kappa_score,
)

def train_model(excel_file:str, test_size:float=0.1, n_estimators:int=100)-> tuple:
    """
    Realiza o treinamento de um modelo para um classficador RandomForest
    Args:
        excel_file::str: Nome do dataset.xlsx que será usada para treinar o modelo
        test_size::float: Quantidade de dados destinados para os testes, onde o restante será destinado para treino.
        n_estimators::int: Número de estimadores (árvores de decisão) que serão utilizados no modelo de RandomForest.

    Returns:
        model, accuracy, precision, recall, f1, kappa::tuple: 
        Tupla que contém, em ordem, o modelo treinado, a acurácia (que expressa a precisão das previsões), a precisão (indicando a proporção de previsões positivas corretas), o recall (que representa a capacidade do modelo de identificar corretamente instâncias positivas), o F1-score (uma média harmônica entre precisão e recall) e o coeficiente Kappa de Cohen (uma medida que avalia a concordância entre as previsões do modelo e os valores reais, ajustada para o acaso). 
    """
    
    data_frame = read_excel(excel_file)

    # Seleciona colunas relevantes para análise
    selected_columns = [
        'Media_Canal_R',
        'Media_Canal_G',
        'Media_Canal_B',
        'Mediana_Canal_R',
        'Mediana_Canal_G',
        'Mediana_Canal_B',
        'Desvio_Canal_R',
        'Desvio_Canal_G',
        'Desvio_Canal_B',
    ]

    
    features = data_frame[selected_columns]
    target = data_frame['Vermifuga']

    # Divide os dados em conjuntos de treino e teste
    train_data, test_data, train_target, test_target = train_test_split(
        features, target, test_size=test_size, random_state=0
    )

    model = RF(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=0,
        max_depth=8,
        max_features=4,
        bootstrap=True,
    )
    model.fit(train_data, train_target)


    predictions = model.predict(test_data)


    accuracy = f'{(accuracy_score(test_target, predictions) * 100):.2f}'
    precision = f'{(precision_score(test_target, predictions) * 100):.2f}'
    recall = f'{(recall_score(test_target, predictions) * 100):.2f}'
    f1 = f'{(f1_score(test_target, predictions) * 100):.2f}'
    kappa = f'{(cohen_kappa_score(test_target, predictions) * 100):.2f}'

    return model, accuracy, precision, recall, f1, kappa


def save_model(model:RF, model_name:str)->None:
    """
    Salva um modelo RandomForestClassifer em um arquivo.pkl 
    
    Args:
        model::RF: Modelo que sera salvo
        model_name::str: Nome do arquivo que será gerado
        
    Return:
        None
    """ 

    with open(f'{model_name}.pkl', 'wb') as file:
        dump(model, file)

def loadModel(name_model='Modelo.pkl'):
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

def useModel(modelo, path):
    
    novo_dado = extract_one(path=path)

    previsao = modelo.predict(novo_dado)

    return bool(previsao)
    
            

if __name__ == '__main__':
    # Avalia o modelo com o arquivo de dados 'FAMACHA_SEGMENTADA'
    model, accuracy, precision, recall, f1, kappa = train_model('FAMACHA_SEGMENTADA')

    # Imprime as métricas de avaliação
    print('Acurácia = ', accuracy)
    print('Precisão = ', precision)
    print('Recall = ', recall)
    print('F1-Score = ', f1)
    print('Kappa = ', kappa)
