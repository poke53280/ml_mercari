
import pandas as pd
import time






DATA_DIR = "C:\\Users\\T149900\\ml_mercari\\"


train = pd.read_table(DATA_DIR + "train.tsv");

test = pd.read_table(DATA_DIR + "test.tsv");

train['item_description'].fillna(value='missing', inplace=True)
test['item_description'].fillna(value='missing', inplace=True)


l = ['must watch. Good acting', 'average movie. Bad acting', 'good movie. Good acting', 'pathetic. Avoid', 'avoid']
 
df = pd.DataFrame(l, columns=['description'])
 
 
from sklearn.feature_extraction.text import CountVectorizer    
word_vectorizer = CountVectorizer(ngram_range=(1,2), analyzer='word')
 
v = word_vectorizer.fit(df['description'])
 
import re
line = " I am having a very nice day."
count = len(re.findall(r'\w+', line))
 
f = v.get_feature_names()
 
q = []
 
for line in f:
    count = len(re.findall(r'\w+', line))
    q.append(count)
 
w = 90
 
test = "movie bad acting good"
 
t_l = test.split()
 
idx = 0
max_gram = 2
 
while idx < len(t_l):
 
    this_l = t_l[idx:idx + max_gram]
 
    str = " ".join(this_l)
 
    print("Testing: " + str)
 
 
    hit1 = " ".join(this_l[:1]) in f
 
    if (idx < len(t_l) -1):
        hit2 = " ".join(this_l[:2]) in f
    else:
        hit2 = False
 
    if hit2:
        print("Found 2-gram")
        idx = idx + 2
    elif hit1:
        print("Found 1-gram")
        idx = idx + 1
    else:
        print("Nothing found")
        idx = idx + 1
 
w = 90
 
l = ["my", "very", "big", "dictionary", "is", "right", "here"]
 
dict = {}
 
 
#1-gram
for x in l:
    key = x
 
    if key in dict and not (dict[key] is None):
        dict[key]= dict[key] + 1
    else:
        dict[key] = 1
 
    d.append(key)
 
 
w = 90
 
for i, j in zip(l, l[1:]):
    key = i + " " + j
 
    if key in dict and not (dict[key] is None):
        dict[key]= dict[key] + 1
    else:
        dict[key] = 1
 
 
w = 90
 
for i, j, k in zip(l, l[1:], l[2:]):
    key = i + " " + j + " " + k
 
    if key in dict and not (dict[key] is None):
        dict[key]= dict[key] + 1
    else:
        dict[key] = 1
 
w = 90




def analyze_run_data():
    f = open(DATA_DIR + "rundata.txt")
    s = f.read()
    f.close()
    l = s.split('\n')
    
    for x in l:
        print(x)





import nltk

def get_nouns(in_str, type_list):
    tokens = nltk.word_tokenize(in_str)
# https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
    is_noun = lambda pos: pos[:2] in type_list 
    
    tagged = nltk.pos_tag(tokens)

    nouns = [word for (word, pos) in tagged if is_noun(pos)] 

    if len(nouns) > 0:
        return " ".join(str(x) for x in nouns)
    else:
        return "none"



print(nltk.pos_tag(nltk.word_tokenize(df.item_description.values[200])))

w = 90

type_list = ['CD', 'JJ', 'LS', 'NN','RB', 'VB']

def noun_ify(series, type_list):
    l = series.values
    n2 = []

    for x in l:
        q = get_nouns(x, type_list)
        n2.append(q)

    return pd.Series(n2)

w = 90


import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])

nlp = spacy.load("en")

def nounify(series):
    l = series.values
    n2 = []

    for x in l:
        q = get_nouns(x)
        n2.append(q)

    return pd.Series(n2)


from spacy.pipeline import Tagger


texts = [u'One doc', u'...', u'Lots of docs']
tagger = Tagger(nlp.vocab)
for doc in tagger.pipe(texts, batch_size=50):
    pass



doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

nouns = []



 nlp.remove_pipe('ner')
 nlp.remove_pipe('parser')

 nlp.pipe_names





def get_nouns_spacy(str):
    doc = nlp(str)

    l = []

    for token in doc:
        if ( (token.pos_ == 'PROPN') | (token.pos_ == 'NOUN')):
            l.append(token.lemma_)

    return l


df = 324


def get_nouns_timed(list):
    s = ".".join((str(x) for x in list))

    start_time = time.time()

    q = get_nouns_spacy(s)

    print("get_nouns_timed [{:5.1f}] s".format( time.time() - start_time))

    return q


print("XXX")



def nounify_spacy(list):
    
    n2 = []

    nRows = len(list)

    nCurrentRow = 0

    start_time = time.time()

    for x in list:
        q = get_nouns_spacy(x)
        n2.extend(q)

        if (round(100 * nCurrentRow/ nRows) != round (100 * (nCurrentRow +1)/ nRows)):
            print('{:5.1f}  [{:5.1f}] s '.format(100 * nCurrentRow/ nRows, time.time() - start_time))

        nCurrentRow = nCurrentRow + 1

    return n2



print("XXX")


def get_list_of_words(series):
    cv = CountVectorizer(ngram_range=(1, 2))
    cv.fit(corpus_data)
    l = cv.get_feature_names()

w = 99
#len train mercari 162032




def list_from_series(series):
    out = []
    l = series.values

    for x in l:
        out.append(x)

    return out




print("XXX")

"""From list to string"""

s = ".".join((str(x) for x in q))


q = get_nouns_spacy(s)




q = nounify(train.name)
train.name = q

print("Nouning test name...")

q = nounify(test.name)
test.name = q


print("Nouning train item description...")

q = nounify(train.item_description)
train.item_description = q

print("Nouning test item description...")

q = nounify(test.item_description)
test.item_description = q

del q


TEXT_DIR = "C:\\Users\\T149900\\Documents\\Visual Studio 2017\\Projects\\ml_mercari\\"

text_files = [TEXT_DIR + "xeno.txt", 
              TEXT_DIR + "gt.txt"]



j = 90

from difflib import SequenceMatcher

def similar(a,b):
    return SequenceMatcher(None, a, b).ratio()

j = 90

print (similar("Anders", "Andres"))


"""Find similarly named items"""

all = pd.concat([train, test])

shoes = all.loc[all.category_name == 'Women/Shoes/Athletic']

all_items = all.name

l = all_items.tolist()

score = []


test_string = "Superman #17 Nov 92"

test_string = "Sea wees size 0"

for x in l:
    this_score = similar(test_string, x)
    score.append(this_score)

i = sorted(range(len(score)),key=lambda x:-score[x])

i = i[0:5]

for x in i:
    print(l[x])


u = 90

def last_word_in_name(df, cat):
    list = []

    if cat == "":
        nameSeries = df.name
    else:
        nameSeries = df.loc[df.category_name == cat].name
    
    a = nameSeries.values

    idx = 0
    while idx < len(a):
        str = a[idx]
        w = str.split()
        last_word = w.pop().lower()[:5]
        list.append(last_word)
        idx += 1

    return list


p = 90

b3 = [val for val in b1 if val in b2]

def to_freq(total_count, values):
    freq = []
    for x in values: freq.append(x/total_count)
    return freq



def get_common_words(l1, l2):

    counts1 = Counter(l1)
    counts2 = Counter(l2)

    labels1, values1 = zip(*counts1.items())
    labels2, values2 = zip(*counts2.items())


    values1 = to_freq(len(l1), values1)
    values2 = to_freq(len(l2), values2)


    indSort = np.argsort(values1)[::-1]
    labels1 = np.array(labels1)[indSort]
    values1 = np.array(values1)[indSort]
    indexes1 = np.arange(len(labels1))

    indSort = np.argsort(values2)[::-1]
    labels2 = np.array(labels2)[indSort]
    values2 = np.array(values2)[indSort]
    indexes2 = np.arange(len(labels2))

q = 324


def text_init():
    train = pd.read_table(DATA_DIR + "train.tsv");
    test = pd.read_csv(DATA_DIR + "test.tsv", sep = "\t", encoding="utf-8", engine = "python");

    full = train.append(test)

    cat0 = "Women/Athletic Apparel/Pants, Tights, Leggings"
    cat1 = "Handmade/Glass/Bottles"
    cat2 = "Women/Tops & Blouses/T-Shirts"
    cat3 = "Women/Tops & Blouses/Tank, Cami"
    cat4 = "Women/Tops & Blouses/Blouse"
    cat5 = "Electronics/Video Games & Consoles/Games"
    cat6 = "Electronics/Cell Phones & Accessories/Cases, Covers & Skins"

    cat7 = "Beauty/Makeup/Face"
    cat8 = "Beauty/Makeup/Lips"
    cat9 = "Beauty/Makeup/Makeup Palettes"


    l1 = last_word_in_name(full, cat1)
    l2 = last_word_in_name(full, cat2)
    l3 = last_word_in_name(full, cat3)
   

    long_string = nameSeries.to_string(index = False)

    corpus_data = [long_string]

    cv = CountVectorizer()

    cv.fit(corpus_data)

    l = cv.get_feature_names()

    vector_corpus = cv.transform(corpus_data)

def text_test(text_str1):
    vector2 = cv.transform([text_str1])

    nz_vec2 = np.nonzero(vector2)

    nz_vec2[1]

    hit_array = nz_vec2[1]


    for idx in np.nditer(hit_array):
        word = l[idx]
        c_c = vector_corpus[0, idx]
        c_t = vector2[0, idx]
        print (word + " " + str(c_c) + " " + str(c_t))


a = 9323
train.loc[train.category_name == "Handmade/Glass/Bottles"]



def plot_test(word_list):
    
    counts = Counter(word_list)
    labels, values = zip(*counts.items())
    indSort = np.argsort(values)[::-1]
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]
    indexes = np.arange(len(labels))
    bar_width = 0.35

    max_num = 50

    indexes = indexes[:max_num]
    values = values[:max_num]
    labels = labels[:max_num]

    plt.bar(indexes, values)

    plt.xticks(indexes + bar_width, labels)
    plt.show()



q = 90

def text_standout():
     tv = TfidfVectorizer(input = 'content', max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')

     cv = CountVectorizer()
     
     long_string = train.name.to_string() + test.name.to_string()

     corpus_data = [long_string]

     cv.fit(corpus_data)

     l = cv.get_feature_names()

     vector_corpus = cv.transform(corpus_data)


    
     text_str1 = "nwt never worn michael kros extra blouse"

     vector2 = cv.transform([text_str1])

     nz_vec2 = np.nonzero(vector2)

     nz_vec2[1]

     hit_array = nz_vec2[1]


     for idx in np.nditer(hit_array):
        word = l[idx]
        c_c = vector_corpus[0, idx]
        c_t = vector2[0, idx]
        print (word + " " + str(c_c) + " " + str(c_t))
        
        
a = 93


def find_name(counter, text):
    sumx = sum(counter.values())

    words = text.split()

    words = [x.lower()[:5] for x in words]



    freq = []

    for x in words:
        f = counter[x]/ sumx
        freq.append(f)

    return freq     
v = 90

"""
list of strings l.
count words
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def count_words(list) :

    cv = CountVectorizer()

    cv.fit(list)


    return 0


v = 90