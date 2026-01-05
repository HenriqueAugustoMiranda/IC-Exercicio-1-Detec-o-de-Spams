import pandas as pnd
import matplotlib.pyplot as mplot
import unicodedata
import nltk
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import TfidfVectorizer


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


def main():
    
    entrada = pnd.read_csv(ENTRADA_PATH, sep='\t', header=None, names=['label', 'text'])
    
    entrada = pre_processamento(entrada)
    entrada = dados_analises(entrada)

    tfidf_mat, vectorizer = tfidf(entrada)
    labels = entrada['label']
    print("Shape da matriz TF-IDF:", tfidf_mat.shape)

main()
