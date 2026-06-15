import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def avaliar(modelo, X_test, y_test, nome_modelo, target):
    # predições
    y_pred = modelo.predict(X_test)

    # métricas
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred, zero_division=0)
    matriz = confusion_matrix(y_test, y_pred)

    # especificidade por classe
    total = matriz.sum()
    especificidades = []
    for i in range(matriz.shape[0]):
        tp = matriz[i, i]
        fp = matriz[:, i].sum() - tp
        fn = matriz[i, :].sum() - tp
        tn = total - tp - fp - fn
        especificidades.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    # prints
    print(f'===== Avaliação: {nome_modelo} → {target} =====')
    print(f'Acurácia global:            {acuracia:.4f}')
    print(f'Especificidade (macro avg): {np.mean(especificidades):.4f}')
    print(f'\nRelatório de classificação:\n{relatorio}')

    # plot da matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão — {nome_modelo} → {target}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(f'./plots/confusion_matrix_{nome_modelo}_{target}.png')
    plt.close()

    return acuracia