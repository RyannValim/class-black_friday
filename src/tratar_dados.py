def tratar_dados(dados):
    colunas_drop = [            # colunas para dropar do treinamento
        'transaction_id',       # id da linha, sem valor preditivo
        'customer_id',          # id do cliente, sem valor preditivo
        'product_id',           # id do produto, sem valor preditivo
        'final_price',          # derivada: original_price * (1 - discount_pct/100), redundante
        'purchase_date'         # 8 datas únicas; is_weekend e is_black_friday já cobrem a informação
    ]

    df = dados.drop(columns=colunas_drop)

    # separação das colunas target
    y_product_category = df['product_category']     # T1: categoria do produto, 10 classes nominais
    y_payment_method = df['payment_method']         # T2: método de pagamento, 6 classes nominais
    y_age_group = df['age_group']                   # T3: faixa etária, 5 classes ordinais

    # drop nas colunas target
    X = df.drop(columns=['product_category', 'payment_method', 'age_group'])

    return X, y_product_category, y_payment_method, y_age_group