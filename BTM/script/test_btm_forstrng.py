
import cleaning_string
import codecs
import pandas as pd
import subprocess
import nltk
import re as regex

import string
import collections
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.wsd import lesk
from nltk.corpus import wordnet
import operator
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import time
import gensim
from gensim import corpora
import csv
import _pickle as cPickle
from nltk.corpus import words as wd


def main():
    string=str(input('enter tweet'))
    string =[string]
    
    file = codecs.open('data.txt', 'w', 'utf-8')
    doc_list=[]
    doc_clean=cleaning_string.get_clean_docs(string)
    for doc in doc_clean:
        document_text=" ".join(doc)
        file.write(document_text+'\n')
    file.close()
    p = subprocess.Popen(["sh", "runExample_1.sh"], shell=True)

    print(p.communicate())
    result_doc=open("C:/Users/brohi/OneDrive/Desktop/BTM-master/output/model/k5.pz_d",'r')
    topic=[]
    for doc in result_doc.readlines():
        doc_dist=doc.split()
        topic.append(doc_dist.index(max(doc_dist)))
    df = pd.DataFrame({'text':string})
    se = pd.Series(topic)
    df['label'] = se

    df.to_csv('output_btm.csv')
    # Data Pre-processing for Sentiment Analysis
    data = TwitterData_Initialize()
    data.initialize("./script/output_btm.csv")
    data.processed_Traindata

    nltk.download('words')
    word_dictionary = list(set(wd.words()))

    for alphabet in "bcdefghjklmnopqrstuvwxyzBCDEFGHJKLMNOPQRSTUVWXYZ":
        word_dictionary.remove(alphabet)
    words = collections.Counter()
    for idx in data.processed_Traindata.index:
        words.update(data.processed_Traindata.loc[idx, "text"])
    stopwords = nltk.corpus.stopwords.words("english")
    whitelist = ["n't", "not"]
    for idx, stop_word in enumerate(stopwords):
        if stop_word not in whitelist:
            del words[stop_word]

    words.most_common(5)

    data = WordList(data)
    data.buildWordlist()
    data = BagOfWords(data)
    bow, labels = data.buildDataModel()
    bow.head(5)
    data = ExtraFeatures()
    data.initialize("./output_btm.csv")
    data.build_features()
    data.cleaningData(DataPreprocessing())
    data.tokenize()
    data.stem()
    data.buildWordlist()
    data_model, labels = data.build_data_model()

    print (data_model)
    # Load Naive-Bayes Library
    with open('../model/NaiveBayesClassifier.pkl', 'rb') as nid:
        nb_loaded = cPickle.load(nid)
    with open('../model/RandomForestClassifier.pkl', 'rb') as rid:
        rf_loaded = cPickle.load(rid)

    result_nb = nb_loaded.predict(data_model)
    print(type(result_nb))
    print("Naive-Bayes Prediction : ", result_nb)
    result_rf = rf_loaded.predict(data_model)
    print("Random Forest Prediction : ", result_rf)
    df_csv = pd.read_csv("./output_btm.csv")
    df_csv['NaiveBayesSentiment'] = pd.DataFrame(result_nb)
    df_csv['RandomForestSentiment'] = pd.DataFrame(result_rf)

    df_csv.to_csv("./output_btm.csv")


# Detecting Emoticons
class EmoticonDetector:
    emoticons = {}

    def __init__(self, emoticon_file="../data/emoticons.txt"):
        from pathlib import Path
        content = Path(emoticon_file).read_text()
        positive = True
        for line in content.split("\n"):
            if "positive" in line.lower():
                positive = True
                continue
            elif "negative" in line.lower():
                positive = False
                continue

            self.emoticons[line] = positive

    def is_positive(self, emoticon):
        if emoticon in self.emoticons:
            return self.emoticons[emoticon]
        return False

    def is_emoticon(self, to_check):
        return to_check in self.emoticons
class TwitterData_Initialize():
    processed_Traindata = []
    wordlist = []
    data_model = None
    data_labels = None

    def initialize(self, csv_file, from_cached=None):
        if from_cached is not None:
            self.data_model = pd.read_csv(from_cached)
            return
        self.processed_Traindata = pd.read_csv(csv_file, usecols=[0, 1])
        self.wordlist = []
        self.data_model = None
        self.data_labels = None

def evaluateFordict(eval_temp,df):
    cout=0
    for index,col in df.iterrows():
            cout += 1
    print ('acc :',(cout/df['negativereason'].value_counts(dropna=True).sum()))
    return cout/df.shape[0]


class DataPreprocessing:
    def iterate(self):
        for preprocessingMethod in [self.replaceProcessedHashtags,
                                    self.removeUrls,
                                    self.removeUsernames,
                                    self.removeElongatedWords,
                                    self.removeNa,
                                    self.replaceSlangWords,
                                    self.removeSpecialChars,
                                    self.removeNumbers]:
            yield preprocessingMethod

    @staticmethod
    def removeByRegex(tweets, regExp):
        tweets.loc[:, "text"].replace(regExp, "", inplace=True)
        return tweets

    def removeUrls(self, tweets):
        return DataPreprocessing.removeByRegex(tweets, regex.compile(r"http.?://[^\s]+[\s]?"))

    def removeNa(self, tweets):
        return tweets[tweets["text"] != ""]

    def removeSpecialChars(self, tweets):
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$"
                                                                                                         "@", "%", "^",
                                                                     "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     "!", "?", ".", "'",
                                                                     "--", "---", "#"]):
            tweets.loc[:, "text"].replace(remove, "", inplace=True)
        return tweets

    def removeUsernames(self, tweets):
        return DataPreprocessing.removeByRegex(tweets, regex.compile(r"@[^\s]+[\s]?"))

    def removeElongatedWords(self, tweets):
        return DataPreprocessing.removeByRegex(tweets, regex.compile(r"(.)\1+', r'\1\1"))

    def removeNumbers(self, tweets):
        # print(tweets)
        return DataPreprocessing.removeByRegex(tweets, regex.compile(r"\s?[0-9]+\.?[0-9]*"))

    def replaceSlangWords(self, tweets):
        with open('../data/slang.txt') as file:
            slang_map = dict(map(str.strip, line.partition('\t')[::2])
                             for line in file if line.strip())
            # print(tweets["text"])
            # print("-----------------------------------------END")
            for index, word in tweets['text'].iteritems():
                # print(index)
                for i in word.split():
                    isUpperCase = i.isupper()
                    i = i.lower()
                    if i in slang_map.keys():
                        word = word.replace(i, slang_map[i])
                        tweets.loc[(index), "text"] = word
                if isUpperCase:
                    i = i.upper()
        # print(tweets.loc[:,"text"])
        return tweets

    # print(split_tweets)
    @staticmethod
    def removeDigitsFromHashtag(tag):
        tag = regex.sub(r"\s?[0-9]+\.?[0-9]*", "", tag)
        return tag

    @staticmethod
    def collect_hashtags_in_tweet(wordList):
        hashtags = []
        for word in wordList:
            index = word.find('#')
            if index != -1:
                if word[index + 1:] != '':
                    hashtags.append(word[index + 1:])
        return hashtags

    @staticmethod
    def split_hashtag_to_words_all_possibilities(hashtag):
        all_possibilities = []

        nltk.download('words')
        word_dictionary = list(set(wd.words()))

        split_posibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag) + 1))]
        possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]

        for split_pos in possible_split_positions:
            split_words = []
            word_1, word_2 = hashtag[:len(hashtag) - split_pos], hashtag[len(hashtag) - split_pos:]

            if word_2 in word_dictionary:
                split_words.append(word_1)
                split_words.append(word_2)
                all_possibilities.append(split_words)

                another_round = DataPreprocessing.split_hashtag_to_words_all_possibilities(word_2)

                if len(another_round) > 0:
                    all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                             zip([word_1] * len(another_round), another_round)]
            else:
                another_round = DataPreprocessing.split_hashtag_to_words_all_possibilities(word_2)

                if len(another_round) > 0:
                    all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                             zip([word_1] * len(another_round), another_round)]

        return all_possibilities

    @staticmethod
    def process_all_hashtags_in_tweet(hashtags):
        all_words = []
        for tag in hashtags:
            split_hashtag = DataPreprocessing.split_hashtag_to_words_all_possibilities(
                DataPreprocessing.removeDigitsFromHashtag(tag))
            if split_hashtag:
                all_words = all_words + split_hashtag[0]
            else:
                all_words.append(tag)
        return all_words

    def replaceProcessedHashtags(self, tweets):
        for index, word in tweets['text'].iteritems():
            word = word.split()
            collectHashtags = DataPreprocessing.collect_hashtags_in_tweet(word)
            allHashtags = DataPreprocessing.process_all_hashtags_in_tweet(collectHashtags)
            collectHashtags = ["#" + tag for tag in collectHashtags]
            if allHashtags:
                word = list(set(word) - set(collectHashtags))
                word = word + allHashtags
            word = " ".join(word)
            tweets.loc[(index), "text"] = word
            # print(tweets.loc[(index),"text"])
        # print(tweets)
        return tweets


class CleanTrainingData(TwitterData_Initialize):
    def __init__(self, previous):
        self.processed_Traindata = previous.processed_Traindata
        # self.processed_Testdata = previous.processed_Testdata

    def cleaningData(self, cleaner):
        train = self.processed_Traindata
        # test = self.processed_Testdata

        for cleanerMethod in cleaner.iterate():
            train = cleanerMethod(train)
            # test = cleanerMethod(test)
        self.processed_Traindata = train
        # self.processed_Testdata = test


class TokenizationStemming(CleanTrainingData):
    def __init__(self, previous):
        self.processed_Traindata = previous.processed_Traindata
        # self.processed_TestData = previous.processed_TestData

    def stem(self, stemmer=nltk.PorterStemmer()):
        def stemJoin(row):
            row["text"] = list(map(lambda str: stemmer.stem(str.lower()), row["text"]))
            return row

        self.processed_Traindata = self.processed_Traindata.apply(stemJoin, axis=1)

    def tokenize(self, tokenizer=nltk.word_tokenize):
        def tokenizeRow(row):
            row["text"] = tokenizer(row["text"])
            row["tokenizedText"] = [] + row["text"]
            return row

        self.processed_Traindata = self.processed_Traindata.apply(tokenizeRow, axis=1)

def eval(df):
    eval_temp={
    '0'  :['Cancelled Flight'],
    '1' : ['Late Flight'],
    '2' : ['Lost Luggage','Damaged Luggage','Flight Attendant Complaints','Flight Booking Problems'],
    '3' : ['Customer Service Issue'],
    '4' :['Can\'t Tell','longlines']
    }
    evaluateFordict(eval_temp,df)

class WordList(TokenizationStemming):
    def __init__(self, previous):
        self.processed_Traindata = previous.processed_Traindata

    whitelist = ["n't", "not"]
    wordlist = []

    def buildWordlist(self, min_occurrences=3, max_occurences=3000, stopwords=nltk.corpus.stopwords.words("english"),
                      whitelist=None):
        self.wordlist = []
        whitelist = self.whitelist if whitelist is None else whitelist
        import os
        if os.path.isfile('../data/wordlist.csv'):
            word_df = pd.read_csv('../data/wordlist.csv', encoding="ISO-8859-1")
            word_df = word_df[word_df["occurrences"] > min_occurrences]
            self.wordlist = list(word_df.loc[:, "word"])
            return
        words = collections.Counter()
        for idx in self.processed_Traindata.index:
            words.update(self.processed_Traindata.loc[idx, "text"])

        for idx, stop_word in enumerate(stopwords):
            if stop_word not in whitelist:
                del words[stop_word]

        word_df = pd.DataFrame(
            data={"word": [k for k, v in words.most_common() if min_occurrences < v < max_occurences],
                  "occurrences": [v for k, v in words.most_common() if min_occurrences < v < max_occurences]},
            columns=["word", "occurrences"])

        word_df.to_csv("../data/wordlist.csv", index_label="idx", encoding="utf8")
        self.wordlist = [k for k, v in words.most_common() if min_occurrences < v < max_occurences]

main()


class BagOfWords(WordList):
    def __init__(self, previous):
        self.processed_Traindata = previous.processed_Traindata
        self.wordlist = previous.wordlist

    def buildDataModel(self):
        columns = list(
            map(lambda w: w + "_bow", self.wordlist))
        labels = []
        rows = []

        for idx in self.processed_Traindata.index:
            currentRow = []

            tokens = set(self.processed_Traindata.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                currentRow.append(1 if word in tokens else 0)

            rows.append(currentRow)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)

        return self.data_model, self.data_labels


class ExtraFeatures(WordList):
    def __init__(self):
        pass

    def build_data_model(self):
        extra_columns = [col for col in self.processed_Traindata.columns if col.startswith("number_of")]
        columns = extra_columns + list(
            map(lambda w: w + "_bow", self.wordlist))

        labels = []
        rows = []
        for idx in self.processed_Traindata.index:
            current_row = []

            for _, col in enumerate(extra_columns):
                current_row.append(self.processed_Traindata.loc[idx, col])

            # adding bag-of-words
            tokens = set(self.processed_Traindata.loc[idx, "text"])
            for _, word in enumerate(self.wordlist):
                current_row.append(1 if word in tokens else 0)

            rows.append(current_row)

        self.data_model = pd.DataFrame(rows, columns=columns)
        self.data_labels = pd.Series(labels)

        return self.data_model, self.data_labels

    def add_column(self, column_name, column_content):
        self.processed_Traindata.loc[:, column_name] = pd.Series(column_content, index=self.processed_Traindata.index)

    def build_features(self):
        def count_by_lambda(expression, word_array):
            return len(list(filter(expression, word_array)))

        def count_occurences(character, word_array):
            counter = 0
            for j, word in enumerate(word_array):
                for char in word:
                    if char == character:
                        counter += 1
            return counter

        def count_interjections(wordArray):
            interjections = []
            interjectionCount = 0
            with open('../data/interjections.txt') as file:
                interjections = file.read().splitlines()
            for word in wordArray:
                if word in interjections:
                    interjectionCount += 1
            return interjectionCount

        def count_by_regex(regex, plain_text):
            return len(regex.findall(plain_text))

        self.add_column("splitted_text", map(lambda txt: txt.split(" "), self.processed_Traindata["text"]))

        # Number of uppercase words
        uppercase = list(map(lambda txt: count_by_lambda(lambda word: word == word.upper(), txt),
                             self.processed_Traindata["splitted_text"]))
        self.add_column("number_of_uppercase", uppercase)

        # number of !
        exclamations = list(map(lambda txt: count_occurences("!", txt),
                                self.processed_Traindata["splitted_text"]))
        self.add_column("number_of_exclamation", exclamations)

        # number of ?
        questions = list(map(lambda txt: count_occurences("?", txt),
                             self.processed_Traindata["splitted_text"]))
        self.add_column("number_of_question", questions)

        # number of ...
        ellipsis = list(map(lambda txt: count_by_regex(regex.compile(r"\.\s?\.\s?\."), txt),
                            self.processed_Traindata["text"]))
        self.add_column("number_of_ellipsis", ellipsis)

        # number of hashtags
        hashtags = list(map(lambda txt: count_occurences("#", txt),
                            self.processed_Traindata["splitted_text"]))
        self.add_column("number_of_hashtags", hashtags)

        # number of mentions
        mentions = list(map(lambda txt: count_occurences("@", txt),
                            self.processed_Traindata["splitted_text"]))
        self.add_column("number_of_mentions", mentions)

        # number of quotes
        quotes = list(map(lambda plain_text: int(count_occurences("'", [plain_text.strip("'").strip('"')]) / 2 +
                                                 count_occurences('"', [plain_text.strip("'").strip('"')]) / 2),
                          self.processed_Traindata["text"]))
        self.add_column("number_of_quotes", quotes)

        # number of urls
        urls = list(map(lambda txt: count_by_regex(regex.compile(r"http.?://[^\s]+[\s]?"), txt),
                        self.processed_Traindata["text"]))
        self.add_column("number_of_urls", urls)

        # number of positive emoticons
        ed = EmoticonDetector()
        positive_emo = list(
            map(lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and ed.is_positive(word), txt),
                self.processed_Traindata["splitted_text"]))
        self.add_column("number_of_positive_emo", positive_emo)

        # number of negative emoticons
        negative_emo = list(
            map(lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and not ed.is_positive(word), txt),
                self.processed_Traindata["splitted_text"]))
        self.add_column("number_of_negative_emo", negative_emo)

        # number of interjections
        interjections = list(map(lambda txt: count_interjections(txt),
                                 self.processed_Traindata["splitted_text"]))
        self.add_column("number_of_interjections", interjections)