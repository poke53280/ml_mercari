
isHome = False

if isHome:
    pass
else:
    import multiprocessing as mp

import pandas as pd
from time import time
from scipy.sparse import csr_matrix, hstack
import os
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
import numpy as np
import gc
from sklearn.base import BaseEstimator, TransformerMixin
import re
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import psutil

import lightgbm as lgb
from scipy import sparse
from sklearn.model_selection import train_test_split


if isHome:
    INPUT_PATH = "C:\\Users\\T149900\\ml_mercari\\"
else:
    INPUT_PATH = r'../input'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['JOBLIB_START_METHOD'] = 'forkserver'

def TXTP_rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


"""c"""

def dameraulevenshtein(seq1, seq2):
   
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            # separate by words by non-alphabetical characters
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        # print("total words processed: %i" % total_word_count)
        # print("total unique words in corpus: %i" % unique_word_count)
        # print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        # print("  edit distance for deletions: %i" % self.max_edit_distance)
        # print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        """return list of suggested corrections for potentially incorrectly
           spelled word"""
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))
                    # early exit
                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        # outlist = sorted(as_list, key=lambda (term, (freq, dist)): (dist, -freq))
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

        '''
        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]  
        '''

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field, start_time=time()):
        self.field = field
        self.start_time = start_time

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        # print(f'[{time()-self.start_time}] select {self.field}')
        dt = dataframe[self.field].dtype
        if is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]


class DropColumnsByDf(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self

    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]


def get_rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


def split_cat(text):
    try:
        cats = text.split("/")
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return 'other', 'other', 'other', 'other/other'


def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    #print(f"Before empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    #print(f"After empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()
    print (str(psutil.virtual_memory().percent) + "%")


def preprocess_regex(dataset, start_time=time()):
    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'
    karats_repl = r'\1k\4'

    unit_regex = r'(\d+)[\s-]([a-z]{2})(\s)'
    unit_repl = r'\1\2\3'

    dataset['name'] = dataset['name'].str.replace(karats_regex, karats_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(karats_regex, karats_repl)
    # print(f'[{time() - start_time}] Karats normalized.')

    dataset['name'] = dataset['name'].str.replace(unit_regex, unit_repl)
    dataset['item_description'] = dataset['item_description'].str.replace(unit_regex, unit_repl)
    # print(f'[{time() - start_time}] Units glued.')


def preprocess_pandas(dataset, start_time=time()):

    dataset['has_category'] = (dataset['category_name'].notnull()).astype('category')
    # print(f'[{time() - start_time}] Has_category filled.')

    dataset['category_name'] = dataset['category_name'].fillna('other/other/other').str.lower().astype(str)

    dataset['general_cat'], dataset['subcat_1'], dataset['subcat_2'], dataset['gen_subcat1'] = zip(*dataset['category_name'].apply(lambda x: split_cat(x)))
    
    dataset['has_brand'] = (dataset['brand_name'].notnull()).astype('category')

    dataset['gencat_cond'] = dataset['general_cat'].map(str) + '_' + dataset['item_condition_id'].astype(str)
    dataset['subcat_1_cond'] = dataset['subcat_1'].map(str) + '_' + dataset['item_condition_id'].astype(str)
    dataset['subcat_2_cond'] = dataset['subcat_2'].map(str) + '_' + dataset['item_condition_id'].astype(str)

    dataset['name'] = dataset['name'].fillna('').str.lower().astype(str)

    dataset['brand_name'] = dataset['brand_name'].fillna('').str.lower().astype(str)
    
    dataset['item_description'] = dataset['item_description'].fillna('').str.lower().replace(to_replace='No description yet', value='')
    # print(f'[{time() - start_time}] Missing filled.')

    preprocess_regex(dataset, start_time)

    brands_filling(dataset)
    # print(f'[{time() - start_time}] Brand name filled.')

    dataset['name'] = dataset['name'] + ' ' + dataset['brand_name']
    # print(f'[{time() - start_time}] Name concancenated.')

    dataset['item_description'] = dataset['item_description'] + ' ' + dataset['name'] + ' ' + dataset['subcat_1'] + ' ' + dataset['subcat_2'] + ' ' + dataset['general_cat'] + ' ' + dataset['brand_name']
    # print(f'[{time() - start_time}] Item description concatenated.')

   


def intersect_drop_columns(train: csr_matrix, valid: csr_matrix, min_df=0):
    t = train.tocsc()
    v = valid.tocsc()
    nnz_train = ((t != 0).sum(axis=0) >= min_df).A1
    nnz_valid = ((v != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_valid
    res = t[:, nnz_cols], v[:, nnz_cols]
    return res

def create_vectorizer_with_meta_data():

    stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this','i', 'so', 'its', 'am', 'are'])
    
    v = FeatureUnion([
        ('name', Pipeline([
            ('select', ItemSelector('name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 2),
                n_features= 75000,
                norm='l2',
                min_df =10,
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('category_name', Pipeline([
            ('select', ItemSelector('category_name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(2, 3),
                token_pattern='.+',
                tokenizer=split_cat,
                n_features=2 ** 27,
                min_df =10,
                norm='l2',
                lowercase=False
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('brand_name', Pipeline([
            ('select', ItemSelector('brand_name', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=10,
                lowercase=False
            )),
        ])),
        ('gencat_cond', Pipeline([
            ('select', ItemSelector('gencat_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_1_cond', Pipeline([
            ('select', ItemSelector('subcat_1_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_2_cond', Pipeline([
            ('select', ItemSelector('subcat_2_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('has_brand', Pipeline([
            ('select', ItemSelector('has_brand', start_time=start_time)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])),
        ('shipping', Pipeline([
            ('select', ItemSelector('shipping', start_time=start_time)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])),
        ('item_condition_id', Pipeline([
            ('select', ItemSelector('item_condition_id', start_time=start_time)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])),
        ('item_description', Pipeline([
            ('select', ItemSelector('item_description', start_time=start_time)),
            ('hash', HashingVectorizer(
                ngram_range=(1, 3),
                n_features=150000,
                dtype=np.float32,
                norm='l2',
                lowercase=False,
                stop_words=stopwords,
                max_df = 0.5,
                min_df = 10

            )),
            ('drop_cols', DropColumnsByDf(min_df=2)),
        ]))
    ], n_jobs=1)

    return v


def create_vectorizer():

    meta_params = {'name_ngram': (1, 2), 'name_max_f': 75000, 'name_min_df': 10,
                   'category_ngram': (2, 3), 'category_token': '.+', 'category_min_df': 10,
                   'brand_min_df': 10,
                   'desc_ngram': (1, 3), 'desc_max_f': 150000, 'desc_max_df': 0.5, 'desc_min_df': 10}

    stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this', ])
    
    v = FeatureUnion([
        ('name', Pipeline([
            ('select', ItemSelector('name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 2),
                n_features=2 ** 27,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('category_name', Pipeline([
            ('select', ItemSelector('category_name', start_time=start_time)),
            ('transform', HashingVectorizer(
                ngram_range=(1, 1),
                token_pattern='.+',
                tokenizer=split_cat,
                n_features=2 ** 27,
                norm='l2',
                lowercase=False
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('brand_name', Pipeline([
            ('select', ItemSelector('brand_name', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('gencat_cond', Pipeline([
            ('select', ItemSelector('gencat_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_1_cond', Pipeline([
            ('select', ItemSelector('subcat_1_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_2_cond', Pipeline([
            ('select', ItemSelector('subcat_2_cond', start_time=start_time)),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('has_brand', Pipeline([
            ('select', ItemSelector('has_brand', start_time=start_time)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])),
        ('shipping', Pipeline([
            ('select', ItemSelector('shipping', start_time=start_time)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])),
        ('item_condition_id', Pipeline([
            ('select', ItemSelector('item_condition_id', start_time=start_time)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])),
        ('item_description', Pipeline([
            ('select', ItemSelector('item_description', start_time=start_time)),
            ('hash', HashingVectorizer(
                ngram_range=(1, 3),
                n_features=2 ** 27,
                dtype=np.float32,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2)),
        ]))
    ], n_jobs=1)

    return v


def base_validate_run(model, X, y, model_name):
     
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(train_X, train_y)
    y_p = model.predict(valid_X)
    print(model_name + " RMSLE: " + str(TXTP_rmsle(np.expm1(valid_y), np.expm1(y_p))))



def FTRL_train(X, y, isValidate):

    print ("FTRL...")

    

    model = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=X.shape[1], iters=47, inv_link="identity", threads=4)

    if isValidate:
        base_validate_run(model, X, y, "FTRL")
   
    model.fit(X, y)

    return model
   
"""c"""

def FM_FTRL_train(X, y, isValidate):

    print ("FM_FTRL...")

    model = FM_FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=0.1, D=X.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                    D_fm=200, e_noise=0.0001, iters=3, inv_link="identity", threads=4)

    if isValidate:
        base_validate_run(model, X, y, "FM_FTRL")


    model.fit(X, y)

    return model

"""c"""

def Ridge_train(X, y, isValidate):

    print ("RIDGE...")

    model = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=200, normalize=False, tol=0.01)

    if isValidate:
        base_validate_run(model, X, y, "Ridge")
    model.fit(X, y)

    return model

"""c"""

def RidgeSAG_A_train(X, y, isValidate):

    print ("RIDGE SAG _A...")

    model = Ridge(solver="sag", fit_intercept=True, alpha = 3.5, random_state=666)

    if isValidate:
        base_validate_run(model, X, y, "Ridge Sag A")
    model.fit(X, y)

    return model

"""c"""

def RidgeSAG_B_train(X, y, isValidate):

    print ("RIDGE SAG _B...")

    model = Ridge(solver="sag", fit_intercept=True, alpha = 1.5, random_state=666)

    if isValidate:
        base_validate_run(model, X, y, "Ridge Sag B")
    model.fit(X, y)

    return model

"""c"""



def LGB_train_A(X, y, isValidate):

    print ("LGBM A...")
    
    params = {
        'learning_rate': 0.67,
        'application': 'regression',
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'feature_fraction': 0.65,
        'nthread': 4,
        'min_data_in_leaf': 100,
        'max_bin': 31
    }


    if isValidate:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42)

        d_train = lgb.Dataset(train_X, label=train_y)
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        
        watchlist = [d_train, d_valid]

        model_lgbm = lgb.train(params, train_set=d_train, num_boost_round=1200, valid_sets=watchlist, verbose_eval=100, early_stopping_rounds=50)
        preds_lgbm = model_lgbm.predict(valid_X)

        print("LGBM A RMSLE " + str(TXTP_rmsle(np.expm1(valid_y), np.expm1(preds_lgbm))))
       
    
    d_train = lgb.Dataset(X, label=y)
    
    watchlist = [d_train]
   
    model = lgb.train(params, train_set=d_train, num_boost_round=1200, valid_sets=watchlist, verbose_eval=100, early_stopping_rounds=50)

    return model

"""c"""

def LGB_train_B(X, y, isValidate):

    print ("LGBM B...")
    
    params = {
        'learning_rate': 1,
        'application': 'regression',
        'num_leaves': 140,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 2,
        'bagging_fraction': 1,
        'nthread': 4
    }


    if isValidate:
        train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42)

        d_train = lgb.Dataset(train_X, label=train_y)
        d_valid = lgb.Dataset(valid_X, label=valid_y)
        
        watchlist = [d_train, d_valid]

        model_lgbm = lgb.train(params, train_set=d_train, num_boost_round=1800, valid_sets=watchlist, verbose_eval=50, early_stopping_rounds=50)
        preds_lgbm = model_lgbm.predict(valid_X)

        print("LGBM B RMSLE " + str(TXTP_rmsle(np.expm1(valid_y), np.expm1(preds_lgbm))))
       
    
    d_train = lgb.Dataset(X, label=y)
    
    watchlist = [d_train]
   
    model = lgb.train(params, train_set=d_train, num_boost_round=1800, valid_sets=watchlist, verbose_eval=50, early_stopping_rounds=50)

    return model

"""c"""

def getW(lm, X, y):
    p0 = lm[0].predict(X)
    p1 = lm[1].predict(X)
    p2 = lm[2].predict(X)

    w0 = 0.9
    w1 = 0.05
    w2 = 0.05

    p = w0*p0 + w1 * p1 + w2 * p2

    o = TXTP_rmsle(np.expm1(p), np.expm1(y))

    o0 = TXTP_rmsle(np.expm1(p0), np.expm1(y))


def trainOnline(start_time, X, X_category_name, y, isValidate):

    y_pred = []

    lm = trainAllModels(start_time, X, y, isValidate)

    assert (len(lm) > 0)

    gc.collect()
    print (psutil.virtual_memory().percent)

    for m in lm:
        p = m.predict(X)
        y_pred.append(p)

    gc.collect()

    if len(lm) > 1:

        X_s = sparse.csr_matrix(np.column_stack(y_pred))

        X_s = hstack((X_s, X_category_name)).tocsr()

        print ("STACK LGBM ...")
    
        params = {
            'learning_rate': 0.01,
            'application': 'regression',
            'num_leaves': 127,
            'verbosity': 1,
            'metric': 'RMSE',
            'data_random_seed': 1,
            'nthread': 4,
            'max_bin': 255
        }


        if isValidate:
            train_X, valid_X, train_y, valid_y = train_test_split(X_s, y, test_size=0.1, random_state=422)
            d_train = lgb.Dataset(train_X, label=train_y)
            d_valid = lgb.Dataset(valid_X, label=valid_y)
        
            watchlist = [d_train, d_valid]

            # For isMeta false
            model_lgbm = lgb.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist, verbose_eval=200, early_stopping_rounds=50)
            preds_lgbm = model_lgbm.predict(valid_X)

            print("LGBM STACK " + str(TXTP_rmsle(np.expm1(valid_y), np.expm1(preds_lgbm))))
       
    
        d_train = lgb.Dataset(X_s, label=y)
    
        watchlist = [d_train]
   
        # For isMeta false
        model = lgb.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist, verbose_eval=200, early_stopping_rounds=50)

        lm.append(model)

    return lm

def predictWeighted(lm, X_t):
    # Assumed more than one base model, and a stack applied to lm.

    y_out = np.zeros(X_t.shape[0])

    nBaseModels = len(lm)- 1

    for m in lm[:-1]:
        y_p = m.predict(X_t)
        y_out = y_out + (1.0/nBaseModels) * y_p

    return y_out


def predictOnline(lm, X_t, X_cat_name_chunk):

    assert (len(lm) > 0)

    if len(lm) > 1:

        y_pred = []

        for m in lm[:-1]:
            y_pred.append(m.predict(X_t))

        a = 90

        X_s = sparse.csr_matrix(np.column_stack(y_pred))

        X_s = hstack((X_s, X_cat_name_chunk))

        # meta regressor

        y = lm[-1].predict(X_s)

        return y
    else:
        y = lm[0].predict(X_t)

        return y


"""c"""

def trainAllModels(start_time, X, y, isValidate):
    lm = []

    start_time = time()
    lm.append(Ridge_train(X, y, isValidate))
    print(f'[{time() - start_time}] base model done.')
    gc.collect()
    print (psutil.virtual_memory().percent)

    #start_time = time()
    #lm.append(RidgeSAG_A_train(X, y, isValidate))
    #print(f'[{time() - start_time}] base model done.')
    #gc.collect()
    #print (psutil.virtual_memory().percent)

    #start_time = time()
    #lm.append(RidgeSAG_B_train(X, y, isValidate))
    #print(f'[{time() - start_time}] base model done.')
    #gc.collect()
    #print (psutil.virtual_memory().percent)

    #start_time = time()
    #lm.append(LGB_train_A(X, y, isValidate))
    #print(f'[{time() - start_time}] base model done.')
    #gc.collect()
    #print (psutil.virtual_memory().percent)

    #start_time = time()
    #lm.append(LGB_train_B(X, y, isValidate))
    #print(f'[{time() - start_time}] base model done.')
    #gc.collect()
    #print (psutil.virtual_memory().percent)


    #start_time = time()
    #lm.append(FTRL_train(X, y, isValidate))
    #print(f'[{time() - start_time}] base model done.')
    #gc.collect()
    #print (psutil.virtual_memory().percent)


    print("Done base models.")
    return lm


"""c"""


if __name__ == '__main__':

    if isHome:
        pass
    else:
        mp.set_start_method('forkserver', True)

    start_time = time()

    train = pd.read_table(os.path.join(INPUT_PATH, 'train.tsv'),  engine='c', dtype={'item_condition_id': 'category',  'shipping': 'category'}  )

    train = train[train.price > 3.0].reset_index(drop=True)

    CHUNK_SIZE = 350000
    isValidate = False
    isOneChunkOnly = False
    nClip = 0
    train = train[:10000]
    isMeta = True

    print('Train shape with price >= 3 ', train.shape)
   
    y_train = np.log1p(train["price"])
    
    train.drop(['price', 'train_id'], axis=1, inplace=True)
   
    print('Train shape: ', train.shape)

    preprocess_pandas(train)

    s = train.category_name.astype('category')

    lbCat = LabelBinarizer(sparse_output=True)

    X_category_name = lbCat.fit_transform(s.cat.codes)
    

    print(f'[{time() - start_time}] Preprocess completed')

    vectorizer = create_vectorizer()
    tfidf_transformer = TfidfTransformer()

    sparse_train = vectorizer.fit_transform(train)
    X_train = tfidf_transformer.fit_transform(sparse_train)

    print(X_train.shape)

    if nClip > 0:
        print("clipping features below " + str(nClip))
        mask = np.array(np.clip(X_train.getnnz(axis=0) - nClip, 0, 1), dtype=bool)
        X_train = X_train[:, mask]
        print(X_train.shape)

    del train
    del sparse_train
    gc.collect()
    

    lm = trainOnline(start_time, X_train, X_category_name, y_train, isValidate)

    del X_train
    del y_train
    gc.collect()

    y_out = []
    test_id_out = []

    reader = pd.read_table(os.path.join(INPUT_PATH, 'test.tsv'), chunksize=CHUNK_SIZE, engine='c', dtype={'item_condition_id': 'category', 'shipping': 'category'} )

    for df_t in reader:
        print("Prediciting " + str(len(df_t)) + " items...")

        test_id_out.extend(df_t.test_id.values)

        df_t.drop(['test_id'], axis=1, inplace=True)

        preprocess_pandas(df_t)

        s = df_t.category_name.astype('category')

        lbCat = LabelBinarizer(sparse_output=True)  

        X_category_name_chunk = lbCat.fit_transform(s.cat.codes)
   
        print (str(psutil.virtual_memory().percent) + "%")
   
        train_on_test = vectorizer.transform(df_t)

        X_t = tfidf_transformer.transform(train_on_test)

        if nClip > 0:
            print("clipping features below " + str(nClip))
            X_t = X_t[:, mask]

        if isMeta:
            y_p = predictOnline(lm, X_t, X_category_name_chunk)
        else:
            y_p = predictWeighted(lm, X_t)

        y_out.extend(y_p)
       
        gc.collect()
        if isOneChunkOnly:
            break

    data_tuples = list(zip(test_id_out,np.expm1(y_out)))

    submission = pd.DataFrame(data_tuples, columns=['test_id','price'])

    submission.loc[submission['price'] < 0.0, 'price'] = 0.0

    if isHome:
        submission.to_csv(os.path.join(INPUT_PATH, "submission.csv"), index=False)
    else:
        submission.to_csv("submission_41.csv", index=False)

    print("All done.")

"""c"""







