from src.carregar_dados import carregar_dados
from src.limpar_dados import limpar_dados
from src.preparar_dados import preparar_dados
from src.treinar_rf import treinar_rf
from src.treinar_knn import treinar_knn
from src.treinar_gboost import treinar_gboost
from src.avaliar import avaliar

def melhor_modelo(target, modelos):
    # modelos = lista de tuplas (nome, acuracia)
    melhor = max(modelos, key=lambda x: x[1])
    print(f'{target}: {melhor[0]} ({melhor[1]:.4f})')

def comparar_modelos(target, modelos):
    # modelos = lista de tuplas (nome, acuracia)
    print(f'\n  {target}:')
    for nome, acc in modelos:
        print(f'      {nome}: {acc:.4f}')

if __name__ == '__main__':
    dados = carregar_dados()

    # limpeza de dados, drop de colunas e separação de dados/targets
    X, y_product_category, y_payment_method, y_age_group = limpar_dados(dados)

    # preparação dos dados, split e preprocessamento
    print('======= DISTRIBUIÇÃO DOS DADOS =======')
    X_train, X_test, y_prod_train, y_prod_test, y_pay_train, y_pay_test, y_age_train, y_age_test = preparar_dados(
        X, y_product_category, y_payment_method, y_age_group
    )

    # treinamento dos modelos
    print('\n======= TREINAMENTO =======')
    print('\n--- Random Forest → product_category ---')
    rf_prod = treinar_rf(X_train, y_prod_train, 'product_category')

    print('\n--- Random Forest → payment_method ---')
    rf_pay = treinar_rf(X_train, y_pay_train, 'payment_method')

    print('\n--- Random Forest → age_group ---')
    rf_age = treinar_rf(X_train, y_age_train, 'age_group')

    print('\n--- KNN → product_category ---')
    knn_prod = treinar_knn(X_train, y_prod_train, 'product_category')

    print('\n--- KNN → payment_method ---')
    knn_pay = treinar_knn(X_train, y_pay_train, 'payment_method')

    print('\n--- KNN → age_group ---')
    knn_age = treinar_knn(X_train, y_age_train, 'age_group')

    print('\n--- GBoost → product_category ---')
    gboost_prod = treinar_gboost(X_train, y_prod_train, 'product_category')

    print('\n--- GBoost → payment_method ---')
    gboost_pay = treinar_gboost(X_train, y_pay_train, 'payment_method')

    print('\n--- GBoost → age_group ---')
    gboost_age = treinar_gboost(X_train, y_age_train, 'age_group')
    
    # avaliação dos modelos
    print('\n======= AVALIAÇÃO =======')
    print('--- Random Forest → product_category ---')
    acc_rf_prod = avaliar(rf_prod, X_test, y_prod_test, 'RandomForest', 'product_category')

    print('\n--- Random Forest → payment_method ---')
    acc_rf_pay = avaliar(rf_pay, X_test, y_pay_test, 'RandomForest', 'payment_method')

    print('\n--- Random Forest → age_group ---')
    acc_rf_age = avaliar(rf_age, X_test, y_age_test, 'RandomForest', 'age_group')

    print('\n--- KNN → product_category ---')
    acc_knn_prod = avaliar(knn_prod, X_test, y_prod_test, 'KNN', 'product_category')

    print('\n--- KNN → payment_method ---')
    acc_knn_pay = avaliar(knn_pay, X_test, y_pay_test, 'KNN', 'payment_method')

    print('\n--- KNN → age_group ---')
    acc_knn_age = avaliar(knn_age, X_test, y_age_test, 'KNN', 'age_group')

    print('\n--- GBoost → product_category ---')
    acc_gboost_prod = avaliar(gboost_prod, X_test, y_prod_test, 'GBoost', 'product_category')

    print('\n--- GBoost → payment_method ---')
    acc_gboost_pay = avaliar(gboost_pay, X_test, y_pay_test, 'GBoost', 'payment_method')

    print('\n--- GBoost → age_group ---')
    acc_gboost_age = avaliar(gboost_age, X_test, y_age_test, 'GBoost', 'age_group')

    # definindo os melhores modelos de cada
    print('\n======= COMPARAÇÃO FINAL =======')
    comparar_modelos('product_category', [
        ('RandomForest', acc_rf_prod),
        ('KNN',          acc_knn_prod),
        ('GBoost',       acc_gboost_prod),
    ])
    comparar_modelos('payment_method', [
        ('RandomForest', acc_rf_pay),
        ('KNN',          acc_knn_pay),
        ('GBoost',       acc_gboost_pay),
    ])
    comparar_modelos('age_group', [
        ('RandomForest', acc_rf_age),
        ('KNN',          acc_knn_age),
        ('GBoost',       acc_gboost_age),
    ])

    print('\n======= MELHORES MODELOS =======')
    melhor_modelo('product_category', [
        ('RandomForest', acc_rf_prod),
        ('KNN',          acc_knn_prod),
        ('GBoost',       acc_gboost_prod),
    ])
    melhor_modelo('payment_method', [
        ('RandomForest', acc_rf_pay),
        ('KNN',          acc_knn_pay),
        ('GBoost',       acc_gboost_pay),
    ])
    melhor_modelo('age_group', [
        ('RandomForest', acc_rf_age),
        ('KNN',          acc_knn_age),
        ('GBoost',       acc_gboost_age),
    ])