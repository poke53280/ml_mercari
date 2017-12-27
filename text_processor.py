
import pandas as pd


DATA_DIR = "C:\\Users\\T149900\\ml_mercari\\"


train = pd.read_table(DATA_DIR + "train.tsv");


import nltk

def get_nouns(in_str):
    tokens = nltk.word_tokenize(in_str)
    is_noun = lambda pos: pos[:2] == 'NN'
    tagged = nltk.pos_tag(tokens)

    nouns = [word for (word, pos) in tagged if is_noun(pos)] 

    if len(nouns) > 0:
        return " ".join(str(x) for x in nouns)
    else:
        return "none"


def nounify(series):
    l = series.values
    n2 = []

    for x in l:
        q = get_nouns(x)
        n2.append(q)

    return pd.Series(n2)



print("Nouning train name...")

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

    