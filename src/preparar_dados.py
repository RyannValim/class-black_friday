from collections import Counter
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.salvar_modelo import salvar_modelo

def preparar_dados(X, y_prod, y_pay, y_age):
    # separação das colunas
    cols_cat = [
        'gender',               # categórica nominal, 3 classes [Male, Female, Other]
        'city',                 # categórica nominal, 10 classes [cidades americanas]
        'customer_segment'      # categórica nominal, 4 classes [Loyal, Returning, New, VIP]
    ]
    
    cols_scaler = [
        'original_price',       # float contínuo, valor original da transação
        'purchase_amount'       # float contínuo, valor real da transação
    ]
    
    cols_ordinais = [
        'discount_pct',         # int ordinal, 10 valores [5,10,15,20,25,30,35,40,50,60]
        'quantity',             # int ordinal, 5 valores [1–5]
        'purchase_hour',        # int ordinal, 24 valores [0–23]
        'is_weekend',           # binária, 0/1
        'is_black_friday'       # binária, 0/1
    ]
    
    # split 70/30 sem stratify (targets já balanceados)
    X_train, X_test, y_prod_train, y_prod_test, y_pay_train, y_pay_test, y_age_train, y_age_test = train_test_split(
        X, y_prod, y_pay, y_age,
        test_size=0.3, random_state=42
    )

    # OHE nas categóricas
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train[cols_cat])
    X_train_cat = ohe.transform(X_train[cols_cat])
    X_test_cat  = ohe.transform(X_test[cols_cat])

    # scaler nas contínuas
    scaler = StandardScaler()
    scaler.fit(X_train[cols_scaler])
    X_train_num = scaler.transform(X_train[cols_scaler])
    X_test_num  = scaler.transform(X_test[cols_scaler])

    # ordinais sem transformação
    X_train_ord = X_train[cols_ordinais].values
    X_test_ord  = X_test[cols_ordinais].values

    # concatena tudo
    X_train = np.hstack([X_train_cat, X_train_num, X_train_ord])
    X_test  = np.hstack([X_test_cat,  X_test_num,  X_test_ord])

    # salva para inferência
    salvar_modelo(ohe, 'OHE_Encoder')
    salvar_modelo(scaler, 'Standard_Scaler')

    print(f'Dados preparados:')
    print(f'    X_train shape: {X_train.shape}')
    print(f'    X_test shape:  {X_test.shape}')

    print(f'\nDistribuição y_prod_train:')
    for classe, quantidade in sorted(Counter(y_prod_train).items()):
        print(f'    {classe}: {quantidade}')

    print(f'\nDistribuição y_pay_train:')
    for classe, quantidade in sorted(Counter(y_pay_train).items()):
        print(f'    {classe}: {quantidade}')

    print(f'\nDistribuição y_age_train:')
    for classe, quantidade in sorted(Counter(y_age_train).items()):
        print(f'    {classe}: {quantidade}')

    return X_train, X_test, y_prod_train, y_prod_test, y_pay_train, y_pay_test, y_age_train, y_age_test