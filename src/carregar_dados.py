import pandas as pd

def carregar_dados():
    return pd.read_csv('./datasets/retail_black_friday_sales_100k.csv', sep=',')