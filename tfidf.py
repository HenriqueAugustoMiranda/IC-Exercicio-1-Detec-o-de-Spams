import matplotlib.pyplot as mplot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import models as m
import grid_search as gs


def tfidf(entrada, min_df= 5, max_df= 0.7, max_features= 2000, ngram_range= (1, 1), sublinear_tf= False):
    
    vectorizer = TfidfVectorizer(
        min_df = min_df,
        max_df=max_df,
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

    KNN_accuracy, KNN_f1 = m.KNN(X_train, X_test, y_train, y_test)
    SVM_accuracy, SVM_f1 = m.SVM(X_train, X_test, y_train, y_test)
    DTC_accuracy, DTC_f1 = m.DTC(X_train, X_test, y_train, y_test)

    labels = ["KNN", "SVC", "Árvore Binária"]
    accuracy = [KNN_accuracy, SVM_accuracy, DTC_accuracy]
    f1 = [KNN_f1, SVM_f1, DTC_f1]

    with open("saida_compare.txt", "w", encoding="utf-8") as f:
        f.write(f"Comparação de Acurácia:\n    knn: {KNN_accuracy}\n    svc: {SVM_accuracy}\n    dtc: {DTC_accuracy}\n\n")
        f.write(f"Comparação de Macro F1:\n    knn: {KNN_f1}\n    svc: {SVM_f1}\n    dtc: {DTC_f1}\n\n")
        f.write("=================================================")

    mplot.figure()
    mplot.bar(labels, accuracy)
    mplot.title("Comparação de Acurácia")
    mplot.ylabel("Acurácia")
    mplot.show()

    mplot.figure()
    mplot.bar(labels, f1)
    mplot.title("Comparação de Macro F1")
    mplot.ylabel("F1-score")
    mplot.show()