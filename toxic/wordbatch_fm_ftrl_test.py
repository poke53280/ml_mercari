
import pyximport; pyximport.install()
import gc
import time
import numpy as np
import pandas as pd
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
import wordbatch
from wordbatch.extractors import WordBag, WordHash, WordSeq
from wordbatch.models import FM_FTRL

import sys
import time

sys.path.append('C:\\tmp2\\FM_FTRL_AVX\\')
from hello9 import FM_FTRL_GITHUB

DATA_DIR = "C:\\Users\\T149900\\ml_mercari\\"


#develop = False
develop= True

def get_mask(X, min_df=2, max_df=100000000):
    t = X.getnnz(axis=0)
    return np.where((t >= min_df) & (t < max_df))[0]

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

class FlatGenerator():
    def __init__(self):
        self.mask= []
        self.wb = None

    def transform(self, df):
        if self.wb is None:
            self.wb= TfidfVectorizer(ngram_range=(1, 1), tokenizer=tokenize,
                             min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                             smooth_idf=1, sublinear_tf=1)
            X = self.wb.fit_transform(df['comment_text'])
        else:
            X = self.wb.transform(df['comment_text'])
        if len(self.mask)==0:
            self.mask= get_mask(X, 3, X.shape[0] // 10)
        X = X[:, self.mask]
        return X

from sklearn.linear_model import LogisticRegression

def main():
    flat_generator= FlatGenerator()

    df_train = pd.read_csv(DATA_DIR + "toxic\\train.csv", engine='c')
    df_train['comment_text'].fillna("unknown", inplace=True)

    if develop:
        df_train, df_valid= train_test_split(df_train, test_size=0.05, random_state=100)
        df_train.reset_index(inplace=True)
        df_valid.reset_index(inplace=True)
    X_train= flat_generator.transform(df_train)
    print(X_train.shape, len(X_train.data))
    if develop:
        X_valid= flat_generator.transform(df_valid)
        mean_val_auc= 0.0
    else:
        df_test= pd.read_csv('../input/test.csv', engine='c')
        df_test['comment_text'].fillna("unknown", inplace=True)
        X_test= flat_generator.transform(df_test)
        submission = pd.read_csv('../input/sample_submission.csv')

    outputs= ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    #model = FM_FTRL(alpha=0.1, beta=0.05, L1=0.0, L2=1.0, D=X_train.shape[1], alpha_fm=0.05, L2_fm=0.01,
    #                 init_fm=0.01, D_fm=4, weight_fm=5.0, e_noise=0.0, iters=50, inv_link="sigmoid", threads=4) #366.019243478775 Mean ROC_AUC: 0.9755822780807716

    model = FM_FTRL_GITHUB(alpha=0.1, beta=0.05, L1=0.0, L2=1.0, D=X_train.shape[1], alpha_fm=0.05, L2_fm=0.01, init_fm=0.01, D_fm=512, weight_fm=5.0, e_noise=0.0, iters=30, inv_link="sigmoid", threads=4, verbose=1, use_baseline=1)
    #model = FM_FTRL_GITHUB(alpha=0.1, beta=0.05, L1=0.0, L2=1.0, D=X_train.shape[1], alpha_fm=0.2, L2_fm=0.01,
                   # init_fm=0.01, D_fm=100, weight_fm=20.0, e_noise=0.000001, iters=50, inv_link="sigmoid",
                  #  threads=1, verbose = 1, use_baseline = 0)  # 366.019243478775 Mean ROC_AUC: 0.9755822780807716 #359.0949101448059 Mean ROC_AUC: 0.9756874348118347Z #594.1383130550385 Mean ROC_AUC: 0.9758896648426597

    """c"""

    nRun = 30

    while nRun > 0:
        print(f"nRun = {nRun}")
        for output in outputs:
            y_train= df_train[output].values

            #model= LogisticRegression(C=4, dual=True)
            print("A")

            T0 = time.time()
            
            model.fit(X_train, y_train)

            T1 = time.time()

            processing_time = T1 - T0

            print(f"Time: {processing_time:.1f}s")

            print("B")
            if develop:
                print("C")
                df_valid[output+"_pred"]= model.predict(X_valid)
                print("D")
                #df_valid[output + "_pred"] = model.predict_proba(X_valid)[:,1]
                val_auc= roc_auc_score(df_valid[output], df_valid[output+"_pred"])
                mean_val_auc+= val_auc/len(outputs)

        nRun = nRun -1

    """c"""
    
if __name__ == '__main__':
    main()