import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.salvar_modelo import salvar_modelo

def treinar_rf(X_train, y_train, nome_modelo):
    # parâmetros a serem utilizados
    n_estimators = [int(x) for x in np.linspace(start=10, stop=300, num=10)]
    criterion = ['gini', 'entropy']
    max_depth = [int(x) for x in np.linspace(start=5, stop=50, num=10)] + [None]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [2, 4]
    max_features = ['sqrt', 'log2']
    bootstrap = [True]
    verbose = 1
    n_jobs = -1
    random_state = 42
    n_iter = 10
    cv = 3
    
    # grid de parâmetros definidos
    rf_grid = {
        'n_estimators': n_estimators,               # indica o tamanho da árvore
        'criterion': criterion,                     # critério a ser usado
        'max_depth': max_depth,                     # profundidade máxima
        'min_samples_split': min_samples_split,     # mínimo de divisão de amostras
        'min_samples_leaf': min_samples_leaf,       # mínimo de nós folhas
        'max_features': max_features,               # máximo de features
        'bootstrap': bootstrap,                     # treina ou não a árvore com reposição do dataset
    }
    
    # instância do RandomForestClassifier
    classificador_rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    
    # instância do RandomizedSearchCV
    hiperparametros_rf = RandomizedSearchCV(
        estimator=classificador_rf,                 # modelo base a ser otimizado
        param_distributions=rf_grid,                # grid de hiperparâmetros para busca aleatória
        n_iter=n_iter,                              # número de combinações aleatórias a testar
        cv=cv,                                      # número de dobras de validação cruzada
        verbose=verbose,                            # quantidade de texto que mostra no print
        n_jobs=n_jobs,                              # utiliza todos os núcleos do processador
        random_state=random_state                   # seed para comparação dos modelos
    )
    
    # treina os hiperparâmetros
    hiperparametros_rf.fit(X_train, y_train)
    
    print('\nMelhor modelo encontrado:')
    for param, valor in hiperparametros_rf.best_params_.items():
        print(f'    {param} = {valor}')

    # pegando o melhor conjunto de estimadores
    melhor_rf = hiperparametros_rf.best_estimator_

    # salva o classificador
    salvar_modelo(melhor_rf, f'RF_Classifier_{nome_modelo}')

    return melhor_rf