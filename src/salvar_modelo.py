from pickle import dump

def salvar_modelo(modelo, nome_modelo):
    dump(modelo, open(f'./models/{nome_modelo}.pkl', 'wb'))