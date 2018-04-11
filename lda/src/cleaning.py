from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import re, string
import pandas as pd
from nltk.stem.snowball import SnowballStemmer




def strip_links(text):
    link_regex =  re.compile('(((http?)|(https?)):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        i=0
        text = text.replace(link[i],'')
        i+=1
    return text


def strip_all_entities(text):
    entity_prefixes = ['@', '#', 'virgin', 'america', 'united', 'delta', 'jetblue', 'southwest']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator, " ")
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                # words.append(" ")
                words.append(word)
            # print(type(''.join(words)))
    return " ".join(words)


def clean(doc):

    stop = set(stopwords.words('english'))

    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    # print(stop_free)
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    # print(punc_free)

    normalized = " ".join(lemma.lemmatize(word) for word in stop_free.split())


    return (normalized)



def get_clean_docs(param):
    if param ==0:
        df = pd.read_csv('resource\\train.csv')
    else:
        df = pd.read_csv('output\\test.csv')
    doc_complete = [row[1] for row in df[df.columns[10+param]].iteritems()]
    nltk.download('stopwords')
    nltk.download('wordnet')
    sp = [strip_links(doc) for doc in doc_complete]
    sp = [strip_all_entities(doc) for doc in sp]

    doc_clean = [clean(doc) for doc in sp]

    doc_clean = [re.sub(r'(.)\1+', r'\1\1', (doc)).split() for doc in doc_clean]
    stemmer2 = SnowballStemmer("english", ignore_stopwords=False)

    for doc in doc_clean:
        if 'u' in doc:
            doc.remove('u')

    print (doc_clean)
    return  doc_clean

get_clean_docs(1)
