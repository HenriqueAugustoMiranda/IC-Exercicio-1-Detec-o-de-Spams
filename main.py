import pandas as pnd
import matplotlib.pyplot as mplot
import unicodedata
import nltk
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from itertools import product


# nltk.download("stopwords")
stopwords = set(sw.words("english"))

ENTRADA_PATH = "smsspamcollection/SMSSpamCollection"


def dados_analises(entrada):

    distribuicao = entrada['label'].value_counts()
    entrada['qtd_palavras'] = entrada['text'].apply(lambda x: len(x.split()))

    distribuicao.plot(kind='bar')
    mplot.title("Distribuição das Classes")
    mplot.show()

    entrada['qtd_palavras'].plot(kind='hist', bins=20)
    mplot.title("Histograma de número de palavras por documento")
    mplot.xlabel("Qtd. de palavras")
    mplot.show()

    print("\nTotal de documentos:", len(entrada))
    print("\nEstatísticas da variável qtd_palavras:")
    print(entrada['qtd_palavras'].describe())

    return entrada


def remover_acentos(texto):

    return ''.join(
        char for char in unicodedata.normalize('NFD', texto)
        if unicodedata.category(char) != 'Mn'
    )


def pre_processamento(entrada):
    
    entrada['text'] = entrada['text'].str.lower()
    entrada['text'] = entrada['text'].apply(remover_acentos)
    entrada['text'] = entrada['text'].str.replace(r"\d+", "<NUM>", regex=True)
    entrada['text'] = entrada['text'].str.replace(r"[^a-z\s]", "", regex=True)
    entrada['text'] = entrada['text'].str.replace(r"\s+", " ", regex=True).str.strip()

    entrada['tokens'] = entrada['text'].apply(lambda x: x.split())
    entrada['tokens'] = entrada['tokens'].apply(lambda x: [p for p in x if p not in stopwords])

    return entrada


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


def KNN(X_train, X_test, y_train, y_test, n_neighbors = 5, weights='distance'):

    KNC = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric='cosine',
        algorithm='brute'
    )
    KNC.fit(X_train, y_train)
    predict = KNC.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict, average="macro")

    return accuracy, f1


def SVM(X_train, X_test, y_train, y_test, C=1, loss='squared_hinge', max_iter=5000):

    svm = LinearSVC(
        C=C,
        loss=loss,
        max_iter=max_iter,
        random_state=42
    )
    svm.fit(X_train, y_train)
    predict = svm.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict, average="macro")

    return accuracy, f1


def DTC(X_train, X_test, y_train, y_test, criterion='gini', max_depth=20, min_samples_split=5,):
    
    dtc = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    dtc.fit(X_train, y_train)
    predict = dtc.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict, average="macro")

    return accuracy, f1


def classify(entrada, X, y):

    knn_parameters, svc_parameters, dtc_parameters, tfidf_parameters = grid_search(entrada, X, y)

    X, vectorizer = tfidf(entrada, **tfidf_parameters)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    KNN_accuracy, KNN_f1 = KNN(X_train, X_test, y_train, y_test, **knn_parameters)
    SVM_accuracy, SVM_f1 = SVM(X_train, X_test, y_train, y_test, **svc_parameters)
    DTC_accuracy, DTC_f1 = DTC(X_train, X_test, y_train, y_test, **dtc_parameters)

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
        'max_iter' : [1000, 3000, 5000]
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
    knn_combination = iterar_grid(knn_grid)
    for knn_params in knn_combination:

        a, f1 = KNN(X_train, X_test, y_train, y_test, **knn_params)
        if f1 > best_knn:
            knn_parameters = knn_params
            best_knn = f1

    # svc
    svc_combination = iterar_grid(svc_grid)
    for svc_params in svc_combination:

        a, f1 = SVM(X_train, X_test, y_train, y_test, **svc_params)
        if f1 > best_svc:
            svc_parameters = svc_params
            best_svc = f1

    # dtc
    dtc_combination = iterar_grid(dtc_grid)
    for dtc_params in dtc_combination:

        a, f1 = DTC(X_train, X_test, y_train, y_test, **dtc_params)
        if f1 > best_dtc:
            dtc_parameters = dtc_params
            best_dtc = f1
    
    # tfidf
    tfidf_combination = iterar_grid(tfidf_grid)
    for tfidf_params in tfidf_combination:

        X, vet = tfidf(entrada, **tfidf_params)
        X_tn, X_ts, y_tn, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)

        k_a, k_f1 = KNN(X_tn, X_ts, y_tn, y_ts, **knn_parameters)
        s_a, s_f1 = SVM(X_tn, X_ts, y_tn, y_ts, **svc_parameters)
        d_a, d_f1 = DTC(X_tn, X_ts, y_tn, y_ts, **dtc_parameters)

        tfidf_score = np.mean([k_f1, s_f1, d_f1])

        if tfidf_score > best_score:
            best_score = tfidf_score
            tfidf_parameters = tfidf_params

    return knn_parameters, svc_parameters, dtc_parameters, tfidf_parameters

def iterar_grid(grid):

    chaves = list(grid.keys())
    valores = list(grid.values())

    for combinacao in product(*valores):
        yield dict(zip(chaves, combinacao))


def main():
    
    entrada = pnd.read_csv(ENTRADA_PATH, sep='\t', header=None, names=['label', 'text'])
    
    entrada = pre_processamento(entrada)
    entrada = dados_analises(entrada)

    X, vectorizer = tfidf(entrada)
    y = entrada['label'].map({'ham': 0, 'spam': 1})
    print("Shape da matriz TF-IDF:", X.shape)

    classify(entrada, X, y)

main()
