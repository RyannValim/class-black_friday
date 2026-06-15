from src.carregar_dados import carregar_dados
from src.tratar_dados import tratar_dados

if __name__ == '__main__':
    dados = carregar_dados()
    X, y_product_category, y_payment_method, y_age_group = tratar_dados(dados)
    
    print(f'[DEBUG] PREVIEW DOS DADOS:\n{dados.head(10)}\n')
    print(f'[DEBUG] TIPO DOS DADOS:\n{dados.dtypes}\n')
    print(f'[DEBUG] X shape: {X.shape}\n')
    print(f'[DEBUG] X colunas: {X.columns.tolist()}\n')
    print(f'[DEBUG] y_product_category:\n{y_product_category.value_counts()}\n')
    print(f'[DEBUG] y_payment_method:\n{y_payment_method.value_counts()}\n')
    print(f'[DEBUG] y_age_group:\n{y_age_group.value_counts()}\n')