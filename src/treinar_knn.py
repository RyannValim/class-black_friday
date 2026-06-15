import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.salvar_modelo import salvar_modelo

def treinar_knn(X_train, y_train, nome_modelo):
    # parâmetros a serem utilizados
    n_neighbors = [int(x) for x in np.arange(start=11, stop=22, step=2)]
    weights = ['uniform']
    metric = ['euclidean', 'manhattan', 'minkowski']
    algorithm = ['ball_tree', 'kd_tree', 'brute', 'auto']
    p = [1, 2, 3]
    verbose = 1
    n_jobs = -1
    random_state = 42
    n_iter = 10
    cv = 3
    
    # grid de parâmetros definidos
    knn_grid = {
        'n_neighbors': n_neighbors,                 # quantos vizinhos consultar para votar na classe
        'weights': weights,                         # vizinhos votam igual/vizinhos mais próximos pesam mais
        'metric': metric,                           # método para medir distância 'euclidean', 'manhattan', 'minkowski'
        'algorithm': algorithm,                     # como o KNN busca os vizinhos internamente
        'p': p                                      # expoente das distâncias
    }
    
    # instância do classificador
    classificador_knn = KNeighborsClassifier()
    
    # instância do RandomizedSearchCV
    hiperparametros_knn = RandomizedSearchCV(
        estimator=classificador_knn,                # modelo base a ser otimizado
        param_distributions=knn_grid,               # grid de hiperparâmetros para busca aleatória
        n_iter=n_iter,                              # número de combinações aleatórias a testar
        cv=cv,                                      # número de dobras de validação cruzada
        verbose=verbose,                            # quantidade de texto que mostra no print
        n_jobs=n_jobs,                              # utiliza todos os núcleos do processador
        random_state=random_state                   # seed para comparação dos modelos
    )
    
    # treina os hiperparâmetros
    hiperparametros_knn.fit(X_train, y_train)
        
    print('\nMelhor modelo encontrado:')
    for param, valor in hiperparametros_knn.best_params_.items():
        print(f'    {param} = {valor}')

    # pegando o melhor conjunto de estimadores
    melhor_knn = hiperparametros_knn.best_estimator_

    # salva o classificador
    salvar_modelo(melhor_knn, f'KNN_Classifier_{nome_modelo}')

    return melhor_knn