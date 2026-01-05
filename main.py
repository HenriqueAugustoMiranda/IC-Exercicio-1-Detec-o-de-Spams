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


def tfidf(entrada):
    
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_features=5000,
        ngram_range=(1,1)
    )

    tfidf_mat = vectorizer.fit_transform(entrada['text'])

    return tfidf_mat, vectorizer


def KNN(X_train, X_test, y_train, y_test):

    KNC = KNeighborsClassifier(n_neighbors=5)
    KNC.fit(X_train, y_train)
    predict = KNC.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict, average="macro")

    return accuracy, f1


def SVM(X_train, X_test, y_train, y_test):

    svm = LinearSVC(random_state=42)
    svm.fit(X_train, y_train)
    predict = svm.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict, average="macro")

    return accuracy, f1


def DTC(X_train, X_test, y_train, y_test):
    
    dtc = DecisionTreeClassifier(
        criterion='gini',
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )

    dtc.fit(X_train, y_train)
    predict = dtc.predict(X_test)

    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict, average="macro")

    return accuracy, f1


def classify(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    KNN_accuracy, KNN_f1 = KNN(X_train, X_test, y_train, y_test)
    SVM_accuracy, SVM_f1 = SVM(X_train, X_test, y_train, y_test)
    DTC_accuracy, DTC_f1 = DTC(X_train, X_test, y_train, y_test)

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


def main():
    
    entrada = pnd.read_csv(ENTRADA_PATH, sep='\t', header=None, names=['label', 'text'])
    
    entrada = pre_processamento(entrada)
    entrada = dados_analises(entrada)

    X, vectorizer = tfidf(entrada)
    y = entrada['label'].map({'ham': 0, 'spam': 1})
    print("Shape da matriz TF-IDF:", X.shape)

    classify(X, y)

main()
