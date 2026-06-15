import pandas as pd
from src.modulo_inferencia import inferir

if __name__ == '__main__':
    # dados para teste — circunstâncias de uma venda fictícia
    venda = [
        'Male',         # gender
        'New York',     # city
        'Loyal',        # customer_segment
        299.99,         # original_price
        30,             # discount_pct
        2,              # quantity
        419.99,         # purchase_amount
        14,             # purchase_hour
        0,              # is_weekend
        1               # is_black_friday
    ]

    # colunas do dataset (mesma ordem do X)
    colunas = [
        'gender', 'city', 'customer_segment',
        'original_price', 'discount_pct', 'quantity',
        'purchase_amount', 'purchase_hour',
        'is_weekend', 'is_black_friday'
    ]

    # construção do dataframe para inferir
    amostra = pd.DataFrame([venda], columns=colunas)

    # modelos treinados por target
    modelos = {
        'product_category': {
            'Random Forest': 'RF_Classifier_product_category',
            'KNN':           'KNN_Classifier_product_category',
            'GBoost':        'GBoost_Classifier_product_category',
        },
        'payment_method': {
            'Random Forest': 'RF_Classifier_payment_method',
            'KNN':           'KNN_Classifier_payment_method',
            'GBoost':        'GBoost_Classifier_payment_method',
        },
        'age_group': {
            'Random Forest': 'RF_Classifier_age_group',
            'KNN':           'KNN_Classifier_age_group',
            'GBoost':        'GBoost_Classifier_age_group',
        },
    }

    print('===== Inferência =====')
    for target, classificadores in modelos.items():
        print(f'\n--- {target} ---')
        for nome, arquivo in classificadores.items():
            predicao, probabilidades, classes = inferir(amostra, arquivo)
            print(f'\n  {nome}:')
            print(f'      Predição:  {predicao}')
            print(f'      Probabilidades por classe:')
            for classe, prob in zip(classes, probabilidades):
                print(f'          {classe}: {prob:.4f}')