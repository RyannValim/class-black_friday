from pickle import load

def carregar_modelo(nome_modelo):
    # carrega o modelo salvo
    modelo = load(open(f'./models/{nome_modelo}.pkl', 'rb'))
    return modelo

def carregar_preprocessador():
    # carrega o OHE e o Scaler salvos
    ohe = load(open('./models/OHE_Encoder.pkl', 'rb'))
    scaler = load(open('./models/Standard_Scaler.pkl', 'rb'))
    return ohe, scaler

def inferir(amostra, nome_modelo):
    ohe, scaler = carregar_preprocessador()
    modelo = carregar_modelo(nome_modelo)

    cols_cat = ['gender', 'city', 'customer_segment']
    cols_scaler = ['original_price', 'purchase_amount']
    cols_ordinais = ['discount_pct', 'quantity', 'purchase_hour', 'is_weekend', 'is_black_friday']

    # aplica as mesmas transformações do treino
    import numpy as np
    amostra_cat = ohe.transform(amostra[cols_cat])
    amostra_num = scaler.transform(amostra[cols_scaler])
    amostra_ord = amostra[cols_ordinais].values
    amostra_proc = np.hstack([amostra_cat, amostra_num, amostra_ord])

    # predição
    predicao = modelo.predict(amostra_proc)[0]
    probabilidades = modelo.predict_proba(amostra_proc)[0]
    classes = modelo.classes_

    return predicao, probabilidades, classes