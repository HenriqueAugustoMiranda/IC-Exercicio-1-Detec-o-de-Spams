import matplotlib.pyplot as mplot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import models as m
import grid_search as gs
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ttest_rel


def tfidf(entrada, min_df= 5, max_df= 0.7, max_features= 2000, ngram_range= (1, 1), sublinear_tf= False):
    
    vectorizer = TfidfVectorizer(
        min_df = min_df,
        max_df = max_df,
        max_features = max_features,
        ngram_range = ngram_range,
        sublinear_tf=sublinear_tf
    )

    tfidf_mat = vectorizer.fit_transform(entrada['text'])

    return tfidf_mat, vectorizer


def classify(entrada, y):

    # knn_parameters, svc_parameters, dtc_parameters, tfidf_parameters = gs.grid_search(entrada, X, y)

    X, vectorizer = tfidf(entrada)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # KNN_accuracy, KNN_f1 = m.KNN(X_train, X_test, y_train, y_test)
    # SVM_accuracy, SVM_f1 = m.SVM(X_train, X_test, y_train, y_test)
    # DTC_accuracy, DTC_f1 = m.DTC(X_train, X_test, y_train, y_test)

    Kfold(X, y)

    # labels = ["KNN", "SVC", "Árvore Binária"]
    # accuracy = [KNN_accuracy, SVM_accuracy, DTC_accuracy]
    # f1 = [KNN_f1, SVM_f1, DTC_f1]

    # with open("saida_compare.txt", "w", encoding="utf-8") as f:
    #     f.write(f"Comparação de Acurácia:\n    knn: {KNN_accuracy}\n    svc: {SVM_accuracy}\n    dtc: {DTC_accuracy}\n\n")
    #     f.write(f"Comparação de Macro F1:\n    knn: {KNN_f1}\n    svc: {SVM_f1}\n    dtc: {DTC_f1}\n\n")
    #     f.write("=================================================")

    # mplot.figure()
    # mplot.bar(labels, accuracy)
    # mplot.title("Comparação de Acurácia")
    # mplot.ylabel("Acurácia")
    # mplot.show()

    # mplot.figure()
    # mplot.bar(labels, f1)
    # mplot.title("Comparação de Macro F1")
    # mplot.ylabel("F1-score")
    # mplot.show()


def Kfold(X, y):
    
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    k_f1_scores = []
    k_acc_scores = []

    s_f1_scores = []
    s_acc_scores = []

    d_f1_scores = []
    d_acc_scores = []

    for tran_i, test_i in skf.split(X, y):

        X_train, X_test = X[tran_i], X[test_i]
        y_train, y_test = y[tran_i], y[test_i]

        k_acc, k_f1 = m.KNN(X_train, X_test, y_train, y_test)
        s_acc, s_f1 = m.SVM(X_train, X_test, y_train, y_test)
        d_acc, d_f1 = m.DTC(X_train, X_test, y_train, y_test)

        k_f1_scores.append(k_f1)
        k_acc_scores.append(k_acc)

        s_f1_scores.append(s_f1)
        s_acc_scores.append(s_acc)

        d_f1_scores.append(d_f1)
        d_acc_scores.append(d_acc)

    with open("saida_kfold.txt", "a")as f:
        f.write("\nKNN - F1 médio:", np.mean(k_f1_scores))
        f.write("KNN - Desvio padrão:", np.std(k_f1_scores))

        f.write("\nSVM - F1 médio:", np.mean(s_f1_scores))
        f.write("SVM - Desvio padrão:", np.std(s_f1_scores))

        f.write("\nDTC - F1 médio:", np.mean(d_f1_scores))
        f.write("DTC - Desvio padrão:", np.std(d_f1_scores))

    t_ks, p_ks = ttest_rel(k_f1_scores, s_f1_scores)
    t_kd, p_kd = ttest_rel(k_f1_scores, d_f1_scores)
    t_sd, p_sd = ttest_rel(s_f1_scores, d_f1_scores)

    with open("saida_kfold.txt", "a")as f:
        f.write("Teste T:")

    interpretar("KNN vs SVM", p_ks)
    interpretar("KNN vs DTC", p_kd)
    interpretar("SVM vs DTC", p_sd)


def interpretar(nome, p):

    alpha = 0.05
    alpha_bonf = alpha / 3

    if p < alpha_bonf:
        with open("saida_kfold.txt", "a")as f:
            f.write(f"{nome}: diferença estatisticamente significativa (p = {p:.5f})")
    else:
        with open("saida_kfold.txt", "a")as f:
            f.write(f"{nome}: modelos estatisticamente equivalentes (p = {p:.5f})")