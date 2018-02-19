


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs

def TXTP_rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def dist_words(w1, w2, e):
    a = (e[w1] + 1) * 0.5
    b = (e[w2] + 1) * 0.5

    o = TXTP_rmsle(a, b)

    print("Distance " + w1 + ", " + w2 +": " + str(o))


"""c"""


sns.set_style("whitegrid")
np.random.seed(0)


DATA_DIR = "C:\\Users\\T149900\\ml_mercari\\"

MAX_NB_WORDS = 100000
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])




print('loading word embeddings...')
embeddings_index = {}
f = codecs.open(DATA_DIR + "toxic\\wiki.simple.vec", encoding="utf8")
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))




train_df = pd.read_csv( DATA_DIR + "toxic\\train.csv")
test_df = pd.read_csv( DATA_DIR + "toxic\\test.csv")

test_df = test_df.fillna('_NA_')

print("num train: ", train_df.shape[0])
print("num test: ", test_df.shape[0])

label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train_df[label_names].values

print (y_train.shape)

train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))

max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)

raw_docs_train = train_df['comment_text'].tolist()
raw_docs_test = test_df['comment_text'].tolist() 

num_classes = len(label_names)


print("pre-processing train data...")
processed_docs_train = []
for doc in tqdm(raw_docs_train):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_train.append(" ".join(filtered))
"""end for"""

processed_docs_test = []
for doc in tqdm(raw_docs_test):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_test.append(" ".join(filtered))
#end for

"""end for"""


tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)

l = processed_docs_train + processed_docs_test

tokenizer.fit_on_texts(processed_docs_train)  #non-leaky

word_index = tokenizer.word_index

print("dictionary size: ", len(word_index))

word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)

word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)


batch_size = 256
num_epochs = 8 

num_filters = 64
embed_dim = 300 
weight_decay = 1e-4


# We can now prepare our embedding matrix limiting to a max number of words:

print('preparing embedding matrix...')

words_not_found = []
words_not_found_index = []

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_dim))

for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
        words_not_found_index.append(i)

print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

# We can finally define the CNN architecture

#CNN architecture
print("training CNN ...")

model = Sequential()


model.add(Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False))



model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]



hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2)


y_test = model.predict(word_seq_test)

submission_df = pd.DataFrame(columns=['id'] + label_names)
submission_df['id'] = test_df['id'].values 
submission_df[label_names] = y_test 
submission_df.to_csv(DATA_DIR + "cnn_fasttext_submission.csv", index=False)


### Gave LB 0.9542
# epoch 5: acc 0.9829 val_acc 0.9793





import spacy

import nltk

nlp = spacy.load('en', disable=['parser', 'ner'])

nlp = spacy.load("en")

from spacy.pipeline import Tagger


texts = [u'One doc', u'...', u'Lots of docs']
tagger = Tagger(nlp.vocab)
for doc in tagger.pipe(texts, batch_size=50):
    pass



doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

nouns = []

nlp.remove_pipe('ner')
nlp.remove_pipe('parser')

nlp.pipe_names;

q = 90

def pos_tag_spacy(str):
    doc = nlp(str)

    for token in doc:
        print(token.pos_  + ", " + token.lemma_ + ", dep=" + token.dep_)


q = 90

def get_pos_words_spacy(txt, type_list):
    doc = nlp(txt)

    l = []

    for token in doc:
        if token.pos_ in type_list:
            l.append(token.lemma_)


    if len(l) > 0:
        return " ".join(str(x) for x in l)
    else:
        return "none"



q = 324

def noun_ify_spacy(series, type_list):
    l = series.values
    n2 = []

    for x in l:
        q = get_pos_words_spacy(x, type_list)
        n2.append(q)

    return pd.Series(n2)

q = 324

s = "moving can not take with me..... Over 200 silver name rare..... Over 100 holographic....Gold master collection box with original 6 holo cards...."

#Count vectorizer, count 200


import re
r = re.compile("([a-zA-Z]+)([0-9]+)")
strings = ['foofo21', 'bar432', 'foobar12345']
print [r.match(string).groups() for string in strings]

#Quantity , and possibly units. Set



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






def get_pos_words_nltk(in_str, type_list):
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

def noun_ify_nltk(series, type_list):
    l = series.values
    n2 = []

    for x in l:
        q = get_pos_words_nltk(x, type_list)
        n2.append(q)

    return pd.Series(n2)

w = 90





def nounify(series):
    l = series.values
    n2 = []

    for x in l:
        q = get_nouns_nltk(x)
        n2.append(q)

    return pd.Series(n2)




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

w = 90

l = []

l.append([3.0,   74.4272287063,  'Reserved Show me your mumu tunic dress', 'Free People', ' Show me your mumu tunic dress'])
l.append([206.0, 24.1283899992,  'BNWT Pink Friday Blue Sherpa', 'PINK', 'Brand new with tags pink friday Sherpa. For kit only.'])
l.append([129.0, 14.8767356555,  'Greenidgal ONLY bundle', 'missing', 'This is specifically for Greenidgal. If you see other things that you want me to add to this bundle, please let me know so I can do so when I ship them out. I will be shipping out tomorrow September 7th. Thanks!'])
l.append([172.0, 20.6358229673,  'Bundle', 'Victoria\'s Secret', 'Good condition Gray and lime green Xs 5 tops 4 bottoms'])
l.append([123.0, 15.105401482,   'Fever Cami for Lovely4','Metal Mulisha', 'NWT'])
l.append([186.0, 23.8170222872,  'Bundle', 'Brandy Melville', 'Reserved'])
l.append([167.0, 22.8545650717,  'Bundle for Alicia', 'Under Armour', 'No description yet'])
l.append([7.0,   50.1166123746,  'Taylor only! Bundle!', 'PINK', 'Bundle for Taylor only!'])
l.append([250.0, 39.6490317348,  'Burberry for Sahara *ON HOLD*', 'Burberry', 'Burberry framed heads print top for @SaharaSmith'])
l.append([76.0,  12.03272516,    'MM CLOTHING', 'missing', 'No description yet'])
l.append([84.0,  13.5049856829,  'Two piece see through', 'missing', 'Top xs bottom fit like a small'])
l.append([164.0, 27.368880402,   'Lularoe', 'missing', '3 XS Irma\'s. All brand new. Never been worn. NEW with tags.'])
l.append([206.0, 35.9685678069,  'SUPER RARE STONE COLD FOX SILK HOLY TUBE', 'Stone Fox Swim', 'Silk floral Holy Tube from Stone Cold Fox. Size 1 works best for US 4. Boned inner bodice, silk floral outer, back zip. New with tags.'])
l.append([169.0, 29.3872433192,  '(XS) Victoria\'s Secret Pink', 'PINK', 'In excellent like new condition no flaws. Loose fit.10 items'])
l.append([65.0,  11.0298154399,  'Pizza Slime Gucci Long Sleeve', 'Gucci', 'Size Small. Worn once Pizza Slime Long Sleeve. Popular black long sleeve as seen on Skrillex and Khloe Kardashian. No longer available on their website. Slight cracking to the graphic on the back of the shirt due to defective print the company used. I\'m selling this shirt because they sent me a new one because of the defective ink. Feel free to make an offer!'])
l.append([195.0, 34.8526599481,  'LuLaRoe', 'LuLaRoe', '3 xxs classic tees 2 xxs irmas 3 xs irmas 1 xxs Randy'])
l.append([114.0, 20.8509916941,  'VICTORIA SECRET PINK SHIRT]', 'PINK', 'Rainbow colors pm black long sleeve shirt,in excellent condition, no flaws.PRICE IS FIRM. This is pam don\'t buy.'])
l.append([6.0,   35.744182611,   'ON Hold !!!! Random VS Pink Collection', 'PINK', 'This is what I call a great collection of smalls !!!! So much fun!! You will receive everything in the picture !! Here is a list of everything you will receive --- • 6 destination YourCityYourPink Stickers - Atlanta , Charlotte , Kansas City , Las Vegas , North Jersey , and San Diego • 2 Free With Purchase only in store Sticker one just says " pink " and the other says " running from my problems " • 1 Victoria Secret Pink KeyChain Giftcard balance [rm] • 1 Victoria\'s Secret pink Iron on Dog Patch • 1 pink nation I\'m in pin / button • 1 Associate/ employee pink pen • 1 travel size Cool and Bright Mist • 2 pink travel size Mood mists in pink Stress no More and Pink Zzzz please • 1 package of pink hair ties and bracelets • And 1 Pink Dog logo gift card holder Can include but may or not extra finds while I\'m cleaning :)'])
l.append([3.0,   19.9855776729,  'Blouses are Reserved for buyer', 'Marshall', 'Two top bundle reserved for a buyer already.'])
l.append([81.0,  14.7514133307,  'PINK LOVE PINK L\/S SHIRT bright pink', 'missing', 'RSVD CLEEKHEATHER..Bright pink with Pink wrote in black bold letters with love pink wrote in script below in script writing. Worn twice,didn\'t think i would be selling and I normally cut tags out of my shirts because of the way some just stick up an out or irritate me. Has not been put in dryer was hung to dry. Worn just around house an also wasn\'t planning on selling any of my clothes until my car broke down an needed money to get fixed or this closet wouldn\'t be here. But I have car payments now so I\'m digging through closet an drawers. Price firm. Thanks for looking.plus was to big.HEATHER #1[rm]#2 [rm]#3[rm]#4[rm].#5 [rm] When bundles are made,it\'s pay when done due to a lot of non payments. Thank you. Adding now.'])
l.append([69.0,  12.4600598307,  'Free people led Zeppelin t shirt', 'missing', 'Worn once. Great t shirt!'])
l.append([222.0, 43.1951783544,  'Jane', 'missing', 'Bundle 226 5/19'])
l.append([81.0,  15.5007073445,  'Palace P45 T-Shirt Grey Marl (Med)', 'Supreme', 'Brand New Palace Skateboards P45 T-Shirt in hand. Bought off the website'])
l.append([110.0, 22.3030934272,  'Bundle for Allanna', 'missing', 'Bundle'])
l.append([93.0,  18.819091727,   'Dkny NWT blouse', 'DKNY', 'Prana yoga long sleeve, white tank, and black beaded tank.'])
l.append([45.0,   9.0699913812,  'Have a nice death.', 'missing', 'Perfect condition.'])


# Bath bombs

b = []

b.append([103.0, 17.6706128964, 'Glitterdollz', '8 items Sex bomb, Golden Wonder bomb, Jester, Butter Bear, Stardust, experimenter, comforter and mistletoe.'])
b.append([124.0, 26.9171780002, 'Lush bundle *RESERVED*', 'All brand new. Includes everything listed in both pictures.'])
b.append([125.0, 28.9232761716, 'Lush Bundle on hold','FOUR brand new reusable bars. A santa in space gift box, a large bubbly shower gel and a night before Christmas gift box'])
b.append([57.0, 15.6284718555, 'Heated Foot Spa', 'Used just a few times - awesome!! Just don\'t have room for it anymore. Purchased brand new from Amazon. Two massage rollers, heating therapy, oxygen bubbles massage, water fall & wave massage. Digital temperature and time control. LED display.'])
b.append([40.0, 10.7961140316, '5 Error 404 Lush Bath Bombs', 'These are the new scent that was sold during black Friday. Wrapped in plastic wrap to keep them fresh.'])
b.append([35.0, 9.52420520714, 'Bath and body works', 'New and never been used'])
b.append([3.0, 12.5056078215, 'For Cassandra', 'Includes One 5oz Bath Bomb Each batch is made from the best organic ingredients and tested for quality •just place in the bath and relax •smells great and makes your skin super soft •Perfect gifts for Valentines Day coming up ✨Baking Soda, Citric Acid, Corn Starch, coconut oil, espom salts, food coloring, glitter, and essential oils (exotic jasmine) ✨tags: bath bomb, gift for her, black magic, Galaxy dust, glitter, LUSH, bath and body works, relax, zen, bath, home accessories, Valentine\'s Day gifts ✨want to bundle more? Just comment and I\'ll make you a listing. 1 for [rm] 2 for [rm] 3 for [rm] 4 for [rm] 5 for [rm] All with free next day shipping and tracking'])
b.append([115.0, 34.2754220551, '4 bath & body works sets', ' Brand new bath & body works set includes 8oz fragrance mist, 8oz body cream, 8.4oz shower gel & 8.4oz body lotion. Pet/smoke free. **price is firm/no trades/no free ship/no holds/will not separate** Cocktail dress Bourbon strawberry & vanilla Poolside coconut colada Golden pear & brown sugar'])
b.append([7.0, 24.0414371843, 'Lush sample empty containers bags bundle', 'Lush sample empty containers & bags bundle. BUNDLES ARE THE BEST☺☺ SAVE $$$ check out my other items in my closet Lush 15 gift shopping bags and gift 10 sample empty containers bundle bags are different size and great condition. The sample containers are good to give as gifts when buyers buy your lush products. presentation in selling you lush products will give you a 5 star rating. these would be great if you are selling Lush items such as perfumes bath bombs and other lush cosmetics. perfect for gift wrapping birthday and holiday present. they are in great condition Lush set Lush cosmetics Lush gifts Lush bath bomb'])

b.append([9.0, 29.4772961577, 'On hold for megalooper', 'Love spell Snowcastle Cherry blossom bubble bar Orange blossom Lux pud Havana sunrise sample'])
b.append([7.0, 22.7906094201, 'On hold for megalooper', 'Brand new easter collection scrub! Shares its scent with honey I washed the kids but is a bit more coconut scented to my nose, similar to Buffy or king of skin :)'])
b.append([35.0, 11.4084350349, 'Full size exotic coconut', 'Brand new never used, full sized products. Shower gel and body splash. Smoke free home. [rm] each or [rm] sold together. I ship on Tuesdays as its my only day off. Pricing is firm! On this item! Free shipping on this item. Selling together, only'])
b.append([65.0, 22.0955372783, 'Lush Kitchen 7 pc. Citrus Spring/Rhea', 'All made fresh in the Kitchen this month Sunny Citrus soap Somewhere Over The Rainbow Soap Over & Over Bomb Ups-A-Daisy bath bomb The Sicilian Mum bomb Your Mother Should Know [rm] plus [rm] shipping'])
b.append([17.0, 5.3549440164, 'PINK bath Bomb', 'Calming Vanilla'])
b.append([9.0, 26.6029652528, 'Body wash bundle', 'This includes (3) 1.2 oz herbal essences body wash (1) 1.7oz old spice Fiji body wash (1) Nivea touch of happiness body wash **NO FREE SHIP**'])
b.append([84.0, 29.9873399684, 'Reserved Lush Bundle', 'RESERVED Free USPS Shipping Brand new and never used 1 x The rough with the smooth 2 x unicorn horn 2 x candy mountain 2 x French kiss 1 x butter bear 1 x Santa\'s belly'])

# Shoes - no or few bundles for inverse check
s = []

s.append([100.0, 20.2953771963, 'Luchesse horn back crocs.', 'missing', 'Women\'s 10. Men\'s 8.5'])
s.append([120.0, 25.1165618361, 'Chanel !!!NEW!!!', 'Chanel', 'NEW CHANEL NO SCRATCHES'])
s.append([55.0, 16.0658097137, 'Vince Verell Slip-On Sneakers 6', 'Vince', 'Great Condition. Only worn 1/2 times. Back of shoe has a little rip from where my sister\'s foot got caught. Not noticeable when wearing. Pictured above. Smoke/Pet Free.'])
s.append([125.0, 40.9778933657, 'Burberry Espadrilles', 'Burberry', 'No description yet'])
s.append([5.0, 16.6341695176, 'Sperry boat shoes Ladies size 8', 'Sperry', 'Ladies animal print Sperry boat shoes Ladies size 8. Gently worn.'])
s.append([9.0, 28.3821428738, 'Pink suede UGGS', 'UGG Australia', 'Great condition, ugg slip ons with pink suede lined with fur inside'])
s.append([9.0, 27.2920805847, 'Pink bow slide on', 'missing', 'This pink bow slip on are brand new never worn. They are a size 7-8 but, can easily fit a 8.5. Very girly and cute'])
s.append([9.0, 24.1096211128, 'Pedro reserved', 'missing', 'In women\'s size 6'])
s.append([81.0, 31.9562287335, 'Black Uggs', 'UGG Australia', 'Black uggs slippers that you can wear anywhere'])

s.append([46.0, 17.9845374972, 'Ariat women\'s Cruiser slip on shoes', 'Ariat', 'Good condition, minimal wear. Very comfortable. Retail for [rm] new.'])
s.append([9.0, 22.1907965682, 'Size 8 Danskin slip on sneakers', 'Danskin', 'Barely worn Size 8 Dansko brand slip ons with memory foam No box'])
s.append([5.0, 12.8517391783, 'moccasins.', 'missing', 'No description yet'])


n = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "elleven", "twelve"]

dsfs
import re

def CutAfter(s):
    
    s = s.lower()

    l = [
        "please",
        "Free shipping",
        "No free shipping",
        "Check out",
        "Will ship",
        "Shipping",
        "Free priority",
        "bundle to save"
        "Ships fast",
        "Freely message",
        "please do not",
        "prices are",
        "Thank you",
        "P.S.",
        "do not buy",
        "please let me",
        "will pack",
        "let me",
        "leave us",
        "all sales are",
        "Price includes" ]

    for x in l:
        x = x.lower()
        s = re.sub(x + ".*$", "___CUT___", s)

    return s

w = 90

def CutSeriesAfter(s):
    for x in s:
        print(CutAfter(x))


w = 90



 #brands I have include....

    # Preprocess:
    # remove n in one
    # two in one
    # Sz.7.5
    # 1960 - 2020 - likely year drop here or in post?

    #number am => drop
    #number pm => drop
    #1st, 2nd

    #stone   - drop (one)

    #...both ...or ...each... X and Y.
    # ...both for or each for... rm or number


    

    #  10 yrs, year years

    # in total ....100 cards...

    # one side, two-sided 

    # all/included/on the in the picture

    # set of... noun + s.

   

    # look for item sum: [('10', 'pajama'), ('3', 'pajama'), ('7', 'pajama')]. Almost certain bundle.
   

    # and H&M, Crazy 8.
 

    # no description yet => na

    

    #model A1218. Worges great,  - '1219. works'

    # 7", 10.1"
    # 1 case is [rm] 2 cases is [rm] 3 cases is [rm] 

    # gen, generation 3rd gen. gen 3. Generation 3. first gen

    # GB, gb, gigs, gig.

 
    # 48MB/s => '48', 'mb'

    # USB 3.0

###############################################################################################
#
#   get_m2_count
#
#

def get_m2_count(l):
    
    n = []
    
    for x in l:
        try:
            num = int(x)
            n.append(num)
        except:
            pass


    a = np.array(n)

    return a.sum()

w = 90


###############################################################################################
#
#   get_m1_count
#
#

def get_m1_count(l):

    num_map = { 'one': 1, 'two': 2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10}

    qty = []
    unit = []

    for x in l:
        qty.append(x[0])
        unit.append(x[1])

    nums = []

    for x in qty:
        if x in num_map:
            print("Found value = " + str(num_map[x]))
            nums.append(num_map[x])
        else:
            try:
                num = int(x)
                nums.append(num)
            except:
                pass

    a = np.array(nums)

    return a.sum()

w = 90



def SingleStringScanner(s):

    s = s.lower()

    #size 10/12
    s = re.sub("\d+\/\d+", "SIZE", s)

    #size 10-12
    s = re.sub("\d+\-\d+", "SIZE", s)


    # Matt23
    s = re.sub("[a-zA-Z]+\d+", "COMBONAME", s)


    s = re.sub("men's \d+\.?\d*", "men's", s)
    s = re.sub("mens \d+\.?\d*", "men's", s)
    s = re.sub("womens \d+\.?\d*", "women's", s)

    s = re.sub("size \d+-\d+", "size", s)
    s = re.sub("size \d+\.?\d*", "size", s)
   
    #print(s)
    letter_digits = "|one|two|three|four|five|six|seven|eight|nine|ten"

    regex_number_and_item = "(\d+\.?\d*" + letter_digits + ")\s*([a-zA-Z]+)"

    regex_number_in_parantheses = "\((\d+)\)"

    regex_word_signal = " (bundle|all|everything|and|collection|set)"

    regex_word_antisignal = " will bundle|each|random|\[rm\]|will combine"

    m1 = re.findall(regex_number_and_item, s)
    m2 = re.findall(regex_number_in_parantheses, s)

    # print (m1)
    # print (m2)

    # remove qty + time, qty + times
 
    l_anti = re.findall(regex_word_antisignal, s)

    # print (l_anti)

    l_pro  =  re.findall(regex_word_signal, s)

    # print (l_pro)

    score_pro = len (l_pro)
    score_anti = len (l_anti)

    len_m1 = len (m1)
    len_m2 = len (m2)

    # print ("score_pro" + str(score_pro) + ", anti = " + str(score_anti) + ", m1 = " + str(len_m1) + ", m2 = " + str(len_m2))

    if (len_m1 == 0 and len_m2 == 0):
        return 1

    if (score_anti > 0 and score_anti >= score_pro):
        return 1

    if (len_m2 >= 1):
        print (m2)
        return get_m2_count(m2)

    if (len_m1 >= 1):
        print (m1)
        return get_m1_count(m1)

    return 1


w = 90

def QtyScanner(l):
    for x in l:
        r = SingleStringScanner(x[3])
        print ("Count = " + str(r))
w = 90

QtyScanner(b)



from nltk.corpus import stopwords
import re

NUM_BRANDS = 4500
NUM_CATEGORIES = 1250




###############################################################################################
#
#   TXTP_split_cat
#

def TXTP_split_cat(text):
    try:
        return text.split("/")
    except:
        return ("No Label", "No Label", "No Label")


###############################################################################################
#
#   TXTP_handle_missing_inplace
#

def TXTP_handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)

w = 90

###############################################################################################
#
#   TXTP_cutting
#

def TXTP_cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'

###############################################################################################
#
#   TXTP_to_categorical
#

def TXTP_to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


###############################################################################################
#
#   TXTP_normalize_text
#

# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def TXTP_normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

w = 90

from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn.linear_model import *
#vct= HashingVectorizer()
#clf= SGDRegressor()

import wordbatch
from wordbatch.models import FTRL
from wordbatch.extractors import WordBag



wb= wordbatch.WordBatch(normalize_text,extractor=(WordBag, {"hash_ngrams":2, "hash_ngrams_weights":[0.5, -1.0],
                                                                    "hash_size":2**23, "norm":'l2', "tf":'log', "idf":50.0}))



clf= FTRL(alpha=1.0, beta=1.0, L1=0.00001, L2=1.0, D=2 ** 25, iters=1)

train_texts= ["Cut down a tree with a herring? It can't be done.", "Don't say that word.", "How can we not say the word if you don't tell us what it is?"]

train_labels= [1, 0, 1]

test_texts= ["Wait! I said it! I said it! Ooh! I said it again!"]

X = wb.transform(train_texts)

clf.fit(X, train_labels)

preds= clf.predict(wb.transform(test_texts))



