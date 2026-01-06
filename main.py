import pandas as pnd
import preprocess as pp
import tfidf as ti

ENTRADA_PATH = "smsspamcollection/SMSSpamCollection"

def main():
    
    entrada = pnd.read_csv(ENTRADA_PATH, sep='\t', header=None, names=['label', 'text'])
    
    entrada = pp.pre_processamento(entrada)
    entrada = pp.dados_analises(entrada)

    # X, vectorizer = ti.tfidf(entrada)
    y = entrada['label'].map({'ham': 0, 'spam': 1})
    # print("Shape da matriz TF-IDF:", X.shape)

    ti.classify(entrada, y)

main()