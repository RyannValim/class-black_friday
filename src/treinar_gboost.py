import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.salvar_modelo import salvar_modelo

def treinar_gboost(X_train, y_train, nome_modelo):
    # parâmetros a serem utilizados
    max_iter = [int(x) for x in np.linspace(50, 200, 10)]       # equivalente ao n_estimators
    learning_rate = [0.01, 0.05, 0.1, 0.2]                      # quanto cada árvore contribui
    max_depth = [3, 4, 5, None]                                 # profundidade máxima
    min_samples_leaf = [10, 20, 30]                             # mínimo de amostras por folha
    l2_regularization = [0.0, 0.1, 1.0]                         # regularização para evitar overfitting
    max_bins = [128, 255]                                       # bins para aproximação dos splits
    verbose = 1
    random_state = 42
    n_iter = 10
    cv = 3

    # grid de parâmetros definidos
    gboost_grid = {
        'max_iter': max_iter,                       # número de iterações boosted
        'learning_rate': learning_rate,             # taxa de aprendizado
        'max_depth': max_depth,                     # profundidade máxima
        'min_samples_leaf': min_samples_leaf,       # mínimo de amostras por folha
        'l2_regularization': l2_regularization,     # regularização L2
        'max_bins': max_bins,                       # bins para histograma
    }
    
    # instância do GradientBoostingClassifier
    classificador_gboost = HistGradientBoostingClassifier(random_state=random_state)
    
    # instância do RandomizedSearchCV
    hiperparametros_gboost = RandomizedSearchCV(
        estimator=classificador_gboost,             # modelo base a ser otimizado
        param_distributions=gboost_grid,            # grid de hiperparâmetros para busca aleatória
        n_iter=n_iter,                              # número de combinações aleatórias a testar
        cv=cv,                                      # número de dobras de validação cruzada
        verbose=verbose,                            # quantidade de texto que mostra no print
        random_state=random_state                   # seed para comparação dos modelos
    )
    
    # treina os hiperparâmetros
    hiperparametros_gboost.fit(X_train, y_train)
    
    print('\nMelhor modelo encontrado:')
    for param, valor in hiperparametros_gboost.best_params_.items():
        print(f'    {param} = {valor}')

    # pegando o melhor conjunto de estimadores
    melhor_gboost = hiperparametros_gboost.best_estimator_

    # salva o classificador
    salvar_modelo(melhor_gboost, f'GBoost_Classifier_{nome_modelo}')

    return melhor_gboost