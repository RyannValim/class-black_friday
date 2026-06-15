
# class-black_friday

Classificador do dataset Black Friday, com módulo de inferência para atribuir dados novos.

## Dataset

[Retail Black Friday Sales Dataset — Kaggle (Noopur Bhatt)](https://www.kaggle.com/datasets/noopurbhatt/retail-black-friday-sales-dataset)

100.000 instâncias, (porém no código foram utilizadas 20.000) transações de varejo capturadas durante o período de Black Friday, cobrindo dados demográficos de clientes, categorias de produtos, preços, descontos e métodos de pagamento.

## Objetivo

O sistema treina classificadores para prever, a partir das circunstâncias de uma venda, três informações distintas:

| Target               | Descrição                   | Classes |
| -------------------- | ----------------------------- | ------- |
| `product_category` | Categoria do produto comprado | 10      |
| `payment_method`   | Forma de pagamento utilizada  | 6       |
| `age_group`        | Faixa etária do comprador    | 5       |

Para cada predição, o sistema retorna o grau de certeza com base nas probabilidades do classificador (`predict_proba`).

## Pipeline

1. Carregamento do CSV (`carregar_dados`)
2. Limpeza: drop de colunas irrelevantes e separação dos três targets (`limpar_dados`)
3. Preparação: split 70/30, OHE nas categóricas, StandardScaler nas contínuas, passthrough nas ordinais (`preparar_dados`)
4. Treinamento de Random Forest, KNN e Gradient Boosting para cada target — 9 modelos no total (`treinar_rf`, `treinar_knn`, `treinar_gboost`)
5. Avaliação com acurácia global, especificidade, sensibilidade, F1-score e matrizes de confusão (`avaliar`)
6. Inferência com amostra fictícia demonstrando os 9 modelos em funcionamento (`inferencia.py`)

## Modelos avaliados

Três algoritmos são comparados para cada target. O melhor por acurácia global é destacado ao final da execução.

* **Random Forest** — conjunto de árvores de decisão com bagging e seleção aleatória de features
* **KNN** — classificação por proximidade com busca de hiperparâmetros via `RandomizedSearchCV`
* **Gradient Boosting Histograma** — boosting sequencial com aproximação por histogramas (`HistGradientBoostingClassifier`), significativamente mais rápido que o GBoost convencional

## Métricas avaliadas

| Métrica               | Descrição                                                   |
| ---------------------- | ------------------------------------------------------------- |
| Acurácia global       | Proporção de predições corretas                           |
| Sensibilidade (Recall) | Capacidade de identificar corretamente cada classe            |
| Especificidade         | Capacidade de rejeitar corretamente o que não é cada classe |
| F1-Score               | Média harmônica entre precisão e sensibilidade             |

As matrizes de confusão são salvas em `plots/` após a avaliação.

## Estrutura do projeto

```
class-black_friday/
├── datasets/
│   └── retail_black_friday_sales_100k.csv
├── models/                          # gerado em runtime, não versionado
├── plots/                           # gerado em runtime, não versionado
├── src/
│   ├── carregar_dados.py
│   ├── limpar_dados.py
│   ├── preparar_dados.py
│   ├── treinar_rf.py
│   ├── treinar_knn.py
│   ├── treinar_gboost.py
│   ├── avaliar.py
│   ├── salvar_modelo.py
│   └── modulo_inferencia.py
├── main.py
├── inferencia.py
├── requirements.txt
└── README.md
```

## Instalação

Clone o repositório e crie o ambiente virtual:

```bash
git clone <url-do-repositorio>
cd class-black_friday
python -m venv .venv
```

Ative o ambiente:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

### Treinamento

```bash
python main.py
```

Executa o pipeline completo. Os modelos treinados são salvos em `models/` e as matrizes de confusão em `plots/`.

### Inferência

```bash
python inferencia.py
```

Demonstra os 9 modelos em funcionamento com uma venda fictícia, exibindo predição e distribuição de probabilidades para cada target. Execute o treinamento antes.

## Dependências

```
pandas
scikit-learn
matplotlib
seaborn
```

## Licença

MIT
