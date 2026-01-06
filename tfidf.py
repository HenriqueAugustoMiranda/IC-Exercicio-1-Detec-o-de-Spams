import matplotlib.pyplot as mplot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import models as m
import grid_search as gs


def tfidf(entrada, min_df = 5, max_df = 1.0, max_features = 5000, ngram_range = (1,1), sublinear_tf = False):
    
    vectorizer = TfidfVectorizer(
        min_df = min_df,
        max_df=max_df,
        max_features = max_features,
        ngram_range = ngram_range,
        sublinear_tf=sublinear_tf
    )

    tfidf_mat = vectorizer.fit_transform(entrada['text'])

    return tfidf_mat, vectorizer


def classify(entrada, X, y):

    knn_parameters, svc_parameters, dtc_parameters, tfidf_parameters = gs.grid_search(entrada, X, y)

    X, vectorizer = tfidf(entrada, **tfidf_parameters)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    KNN_accuracy, KNN_f1 = m.KNN(X_train, X_test, y_train, y_test, **knn_parameters)
    SVM_accuracy, SVM_f1 = m.SVM(X_train, X_test, y_train, y_test, **svc_parameters)
    DTC_accuracy, DTC_f1 = m.DTC(X_train, X_test, y_train, y_test, **dtc_parameters)

    labels = ["KNN", "SVC", "Árvore Binária"]
    accuracy = [KNN_accuracy, SVM_accuracy, DTC_accuracy]
    f1 = [KNN_f1, SVM_f1, DTC_f1]

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