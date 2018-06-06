# http://napitupulu-jon.appspot.com/posts/text-learning-ud120.html


import numpy

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

string1 = "hi Katie the self driving car will be late Best Sebastian"
string2 = "Hi Sebastian the machine learning will be great greate best Katie"

email_list = [string1,string2]

vectorizer.fit(email_list)


bag_of_words = vectorizer.transform(email_list)

print (bag_of_words)

print (vectorizer.vocabulary_.get('great'))

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

stemmer.stem("responsiveness")


stem_rus = SnowballStemmer("russian")

stem_rus.stem("Плотный")

in_data = "Плотный Продам свитер из Англии, фирма Woolovers, 100% хлопок. Не подошел размер. Свитер идет на 56-58 примерно размер. Плотный, не тонкий. Отдаю за свою цену, перезакажу меньший размер."

list = in_data.split(" ")

stem_rus.stem(list[2])


from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import FrenchStemmer

stemmer = FrenchStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

stem_vectorizer = CountVectorizer(analyzer=stemmed_words)
print(stem_vectorizer.fit_transform(['Tu marches dans la rue']))
print(stem_vectorizer.get_feature_names())



data0 = "продам свитер из англии, фирма Wооловерс, 100% хлопок"
data1 = "продам свитер из англии, фирма Wооловерс?"

data3 = "Продаётся Daewoo nexia 1999 год.вып."


from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import RussianStemmer

from nltk.corpus import stopwords



stemmer = RussianStemmer()

#
# Working with respect to examples on
# http://snowball.tartarus.org/algorithms/russian/stemmer.html
#
# stemmer.stem("важных")
# => 'важн
#
# stemmer.stem("падшему")
# => 'падш'
#

stopWords = stopwords.words('russian')

analyzer = CountVectorizer().build_analyzer()


def analyzer2(str):
    l = str.split(" ")
    return l
"""c"""


def stemmed_words(doc_string):
    return (stemmer.stem(w) for w in analyzer2(doc_string))

# Note: stop_words only in play with analyzer='word', according to documentation.

stem_vectorizer = CountVectorizer(analyzer=stemmed_words, stop_words = stopWords)


q = train.description.fillna('')


a = stem_vectorizer.fit_transform(q[1000:1002])

# <3000x11403 sparse matrix of type '<class 'numpy.int64'>'
# with 57844 stored elements in Compressed Sparse Row format>

print(stem_vectorizer.get_feature_names())



#### Spacy

https://github.com/kmike/pymorphy2


from spacy.lang.ru import Russian
# Requires morph2

import spacy


nlp = Russian()  # use directly


txt = train.description[900]


doc = nlp(txt)

for token in doc:
    print(token.text)

"""c"""

==> Not seeing any results.




