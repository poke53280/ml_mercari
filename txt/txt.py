

# 2T Fredag.
# 3T Lørdag. + 1





import spacy

nlp = spacy.load("nb_dep_ud_sm")

doc = nlp(u"Sliter med smerter i ryggen. Deltaker er sykmeldt.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)
"""c"""


##################################################################################
#
#
#    Find 0103-020417
#
#

import re
import pandas as pd

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=200)
np.core.arrayprint._line_width = 480

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

date_0_epoch = pd.to_datetime("211198", errors='raise', format="%d%m%y") - pd.to_datetime("010170", errors='raise', format="%d%m%y")

nEpochBaseline = date_0_epoch.days

input_full =[
    "Utenlandsopphold 1012-211288 , 1012-211388 og 0101-020192",
    "Utenlandsopphold 1112-231288 godkjent,  i 1012-211388 men ikke 0101-020192",
    "Utenlandsopphold 1012-211288 godkjent,  i 1g012-211388 0102-0f20192"
    ]


res_input = []
res_start = []
res_length = []
res_status = []
res_baseline = []

for input in input_full:

    t = re.findall(r"\b\d{4}[- ]\d{6}\b", input)

    t_int_start = []
    t_int_length = []
    t_int_status = []

    for idx, x in enumerate(t):

        ds = re.findall(r"\d+", x)

        startd4 = ds[0]
        endd6 = ds[1]

        startd6 = startd4 + endd6[-2:]

        parseOK = True

        try:
            epoch_day = pd.to_datetime("010170", errors='raise', format="%d%m%y")
            t_from = pd.to_datetime(startd6, errors='raise', format="%d%m%y")
            t_to = pd.to_datetime(endd6, errors='raise', format="%d%m%y")

        except ValueError:
            parseOK = False

        isPositiveDaysOK = False

        if parseOK:
            dDays = (t_to - t_from).days
            isPositiveDaysOK = dDays > 0

        if isPositiveDaysOK:
            t_epoch_start = (t_from - epoch_day).days
            t_int_start.append(str(t_epoch_start))
            t_int_length.append(str(dDays))
            t_int_status.append("True")

            day_start_relative = t_epoch_start - nEpochBaseline

            if day_start_relative >= 0:
                representation = f"i{idx}ahead{day_start_relative}d_{dDays}d"
            else:
                representation = f"i{idx}back{-day_start_relative}d_{dDays}d"

            input = input.replace(x, representation)

        else:
            t_int_start.append(str(0))
            t_int_length.append(str(0))
            t_int_status.append("False")
            input = input.replace(x, f"i{idx}_err")


    if len (t_int_status) > 0:
        t_out_start = ", ".join(t_int_start)
        t_out_length = ", ".join(t_int_length)
        t_out_status = ", ".join(t_int_status)

    else:
        t_out_start = "None"
        t_out_length = "None"
        t_out_status = "None"

    res_input.append(input)
    res_start.append(t_out_start)
    res_length.append(t_out_length)
    res_status.append(t_out_status)

    res_baseline.append(nEpochBaseline)


df = pd.DataFrame({'in': input_full, 'out': res_input, 'start': res_start, 'length': res_length, 'baseline': res_baseline, 'status': res_status})

"""c"""








##################### TLF


def extract_tlf(l):

    l_tot_clean = []
    l_tot_tlf = []


    for t in l:
        m = re.findall(r"\d{2}[ \.-]*\d[ \.-]*\d[ \.-]*\d[ \.-]*\d[ \.-]*\d[ \.-]*\d", t)

        l_tlf = []
        t_cleaned = t

        for idx, x in enumerate(m):
            tlf_raw_match = x
            tlf = x

            tlf = tlf.replace(" ", "")
            tlf = tlf.replace(".", "")
            tlf = tlf.replace("-", "")

            t_cleaned = t_cleaned.replace(tlf_raw_match, f" TLF{idx} ")
            l_tlf.append(tlf)
        

        res = ", ".join(l_tlf)
        # print (f"{t}: => '{t_cleaned}'. Numbers: {res}")

        l_tot_tlf.append(res)

        t_cleaned = t_cleaned.replace("  ", " ")
        t_cleaned = t_cleaned.replace("  ", " ")
        t_cleaned = t_cleaned.replace("  ", " ")
        t_cleaned = t_cleaned.replace(" .", ".")
        l_tot_clean.append(t_cleaned)

    df = pd.DataFrame({'in': l, 'out': l_tot_clean, 'tlf' : l_tot_tlf})

    return df

"""c"""

l = [   "d2d2d 223.34552 d13234523 f, 22 33...44 55",
        "22.33 .44.55..., eller kanskje 343.43453ekl fem",
        "tlf er. 22334455.",
        "prøvde å ringe 223.34552 kl3",
        "223-93132"]


extract_tlf(l)




################################################################################

from gensim.models.wrappers import FastText


import numpy as np
import pandas as pd

DATA_DIR = "C:\\NORDBANK\\"

df = pd.read_csv(DATA_DIR + 'wiki.no.vec', chunksize = 1000, delimiter = ' ')


df_c = pd.DataFrame(df.get_chunk(1500))
df_c = df_c.reset_index()


# Vanlige ord

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import re


def replace(d, replace_these, replace_value):
    l_keys = list (d.keys())
    l_values = list (d.values())

    l_values_new = []

    for w in l_values:
        if w in replace_these:
            l_values_new.append(replace_value)
        else:
            l_values_new.append(w)                

    d = dict(zip(l_keys, l_values_new))
    return d
"""c"""


def recreate(d_reverse, seps, d_idx):

    l_res = []

    for idx, line in enumerate(d_idx):
        l_words = []
        for x in line:
            if x in d_reverse:
                l_words.append(d_reverse[x])
            else:
                l_words.append("UNK")

        s = seps[idx]

        c = l_words + s
        c[::2] = l_words
        c[1::2] = s

        l_res.append("".join(c))
    """c"""

    return l_res
"""c"""


text0 = 'The quick brownfox: Jumped over an elephant in Skien - and went lazy? sheep dog!'
text1 = 'The sheep, also in Skien with a quick monkey and another elephant? went to: sleep...'

data = [text0, text1]

seps = []

for d_t in data:
    seps.append(re.findall(r"[^a-zA-ZæøåÆØÅ0-9]+", d_t))
"""c"""

t = Tokenizer()

t.fit_on_texts(data)

d_idx = t.texts_to_sequences(data)

d = t.word_index

d_reverse = dict(zip(d.values(), d.keys()))

replace_these = ['brownfox', 'sheep', 'elephant', 'monkey', 'dog']
replace_value = "animal"

d_reverse = replace(d_reverse, replace_these, replace_value)

data = recreate(d_reverse, seps, d_idx)

t = Tokenizer()

t.fit_on_texts(data)

d_idx = t.texts_to_sequences(data)

d = t.word_index

d_reverse = dict(zip(d.values(), d.keys()))

# Basic cut

l_reverse_keys = list (d_reverse.keys())
l_reverse_values = list (d_reverse.values())

l_reverse_keys = l_reverse_keys[:7]
l_reverse_values = l_reverse_values[:7]

d_reverse = dict(zip(l_reverse_keys, l_reverse_values))

data = recreate(d_reverse, seps, d_idx)

data



