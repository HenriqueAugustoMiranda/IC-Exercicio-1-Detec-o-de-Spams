from sklearn.model_selection import train_test_split
import numpy as np
from itertools import product
import models as m
import tfidf as ti


def grid_search(entrada, X_base, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)

    tfidf_grid = {
        'min_df': [1, 2, 3, 5],
        'max_df': [0.7, 0.8, 0.9, 1.0],
        'max_features': [2000, 3000, 5000, 8000],
        'ngram_range': [(1,1), (1,2)],
        'sublinear_tf': [True, False]
    }
    tfidf_parameters = None
    best_score = -1

    knn_grid = {
        'n_neighbors' : [3, 5, 7, 9],
        'weights' : ['uniform', 'distance'],
    }
    knn_parameters = None
    best_knn = -1

    svc_grid = {
        'C': [0.1, 1, 10],
        'loss': ['hinge', 'squared_hinge'],
        'max_iter' : [3000, 5000, 8000, 10000]
    }
    svc_parameters = None
    best_svc = -1

    dtc_grid = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20]
    }
    dtc_parameters = None
    best_dtc = -1

    # knn
    print("[GRID] testando KNN")
    knn_combination = iterar_grid(knn_grid)
    for knn_params in knn_combination:

        a, f1 = m.KNN(X_train, X_test, y_train, y_test, **knn_params)
        if f1 > best_knn:
            knn_parameters = knn_params
            best_knn = f1
    print(f"[GRID] melhor pontuacao de f1:{best_knn}\nmelhores parametros encontrados:{knn_parameters}\n=============================================")

    # svc
    print("[GRID] testando SVC")
    svc_combination = iterar_grid(svc_grid)
    for svc_params in svc_combination:

        a, f1 = m.SVM(X_train, X_test, y_train, y_test, **svc_params)
        if f1 > best_svc:
            svc_parameters = svc_params
            best_svc = f1
    print(f"[GRID] melhor pontuacao de f1:{best_svc}\nmelhores parametros encontrados:{svc_parameters}\n=============================================")

    # dtc
    print("[GRID] testando DTC")
    dtc_combination = iterar_grid(dtc_grid)
    for dtc_params in dtc_combination:

        a, f1 = m.DTC(X_train, X_test, y_train, y_test, **dtc_params)
        if f1 > best_dtc:
            dtc_parameters = dtc_params
            best_dtc = f1
    print(f"[GRID] melhor pontuacao de f1:{best_dtc}\nmelhores parametros encontrados:{dtc_parameters}\n=============================================")

    # tfidf
    print("[GRID] testando TF-IDF")
    tfidf_combination = iterar_grid(tfidf_grid)
    for tfidf_params in tfidf_combination:

        X, vet = ti.tfidf(entrada, **tfidf_params)
        X_tn, X_ts, y_tn, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)

        k_a, k_f1 = m.KNN(X_tn, X_ts, y_tn, y_ts, **knn_parameters)
        s_a, s_f1 = m.SVM(X_tn, X_ts, y_tn, y_ts, **svc_parameters)
        d_a, d_f1 = m.DTC(X_tn, X_ts, y_tn, y_ts, **dtc_parameters)

        tfidf_score = np.mean([k_f1, s_f1, d_f1])

        if tfidf_score > best_score:
            best_score = tfidf_score
            tfidf_parameters = tfidf_params
    print(f"[GRID] melhor score:{best_score}\nmelhores parametros encontrados:{tfidf_parameters}\n=============================================")

    return knn_parameters, svc_parameters, dtc_parameters, tfidf_parameters

def iterar_grid(grid):

    chaves = list(grid.keys())
    valores = list(grid.values())

    for combinacao in product(*valores):
        yield dict(zip(chaves, combinacao))