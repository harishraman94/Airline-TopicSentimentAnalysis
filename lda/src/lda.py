import pandas as pd
from src import cleaning
from gensim import corpora
import gensim
import csv
def main():
#    doc_clean=cleaning.get_clean_docs('train.csv')
#    dictionary = corpora.Dictionary(doc_clean)
#    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
#    ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=150)
     #ldamodel = Lda.load("lda")
#   ldamodel.save("output\\lda_10")
    ldamodel=Lda.load('output\\lda_10')
#    df = pd.read_csv('test.csv')

    doc_clean= cleaning.get_clean_docs(1)
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    label_list=[max(
        ldamodel.get_document_topics(doc, minimum_probability=None, minimum_phi_value=None, per_word_topics=False),
        key=lambda item: item[1])[0] for doc in doc_term_matrix]
    topic_count={}
    for label in label_list:
        if label in topic_count:
            topic_count[label]=topic_count[label] + 1
        else:
            topic_count[label]=1

    se = pd.Series(label_list)
    print (topic_count)
#    df = pd.read_csv('Tweets.csv')
    df = pd.read_csv('output\\test.csv')
    df['label'] = se
#    df.to_csv('modified_tweets5.csv')
    df.to_csv('output\\modified_test5.csv')
    topics = ldamodel.show_topics(num_topics=5, num_words=10, log=False, formatted=True)
    with open('output\\topics5_test.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(topics)


if __name__ == '__main__':
    main()
