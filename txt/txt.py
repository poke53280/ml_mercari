

# 2T Fredag.
# 3T Lørdag. + 1  + 1   + 2 + 2 = 9timer
 


# Todo: Incorporate into Infotrygd preprocessing from here:

# 1. Interval extraction and masking.
# 2. Phone number extranction and masking.
# 3. Seperator conservation.
# 4. Word correction
# 5. GML/Stedsnavn-list imported.





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


##################### TLF ##############################


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


############################### SPELL CORRECTION ###########################

import pandas as pd
import numpy as np


class TypoCheck:

    _wordlist = []

    def __init__(self, filename):
        df = pd.read_csv(filename, skiprows=25, encoding = 'latin-1', error_bad_lines = False, sep = '\t', low_memory=False)
        df.columns = ['A', 'B', 'C', 'D', 'E', 'F']
        l = list (df.C)
        s = set(l)
        self._wordlist= list (s)


    def exec(self, test):
        if test in self._wordlist:
            # print("Exact match")
            return test

        start_letters = test[:1]
        
        l = self._wordlist
        result = [i for i in l if str(i).startswith(start_letters)]

        score = []

        n = 0
        for x in result:
            n = n + 1
            s = dameraulevenshtein(test, x)
            score.append(s)

        anScore = np.array(score)

        m = anScore <= 1

        if m.sum() > 1:
            # print("Multiple results, returning input")
            return test

        if m.sum() == 0:
            # print("None close, returning input")
            return test


        iClose = np.where(m)[0][0]

        return result[iClose]


    def process(self, l):
        l_clean = []
        for x in l:
            l_clean.append(self.exec(x))

        return l_clean

"""c"""

t = TypoCheck(DATA_DIR + "\\ordbank_bm\\fullform_bm.txt")

l = ['innsat', 'utemfor', 'utrollig', 'uvillje', 'ubetallelige', 'uvanelig', 'ufetimelig', 'uforenlig', 'utennforstående', 'umennskelig']

t.process(l)


# Check if close to sorted frequency list, before self.

import networkx as nx
import gc
import xml.etree.ElementTree

DATA_DIR = "c:\\NORDBANK"

e = xml.etree.ElementTree.parse(DATA_DIR + "\\basisdata.gml").getroot()

idx = 301

e[idx][0].findall("{http://skjema.geonorge.no/SOSI/produktspesifikasjon/Stedsnavn/5.0}navneobjekttype")

print(xml.etree.ElementTree.tostring(e[idx][0], encoding='utf8').decode('utf8'))





def get_type(e, idx):
    c = e[idx][0].findall("{http://skjema.geonorge.no/SOSI/produktspesifikasjon/Stedsnavn/5.0}navneobjekttype")
    c = c[0]

    return c.text
"""c"""


def get_name(e, idx):

    # print(xml.etree.ElementTree.tostring(e[idx][0], encoding='utf8').decode('utf8'))

    c = e[idx][0].findall("{http://skjema.geonorge.no/SOSI/produktspesifikasjon/Stedsnavn/5.0}stedsnavn")
    c = c[0]
    c = c[0]

    c = c.findall("{http://skjema.geonorge.no/SOSI/produktspesifikasjon/Stedsnavn/5.0}skrivemåte")
    c = c[0]
    c = c[0]

    c = c.findall("{http://skjema.geonorge.no/SOSI/produktspesifikasjon/Stedsnavn/5.0}langnavn")
    c = c[0]

    return c.text
"""c"""

idx = 0

l_words = []
l_type = []

while True:
    try:
        n = get_name(e, idx)
        assert len(n) > 0
        l_words.append(n)

        t = get_type(e, idx)
        assert len(t) > 0
        l_type.append(t)

        idx = idx + 1
    except IndexError:
        print ("Out of range")
        break
        
"""c"""

t = tuple(zip (l_words, l_type))


for w, t in zip (l_words, l_type):
    if w == "Gamlebyen":
        print (t)
"""c"""


# Losing a lot of the many - many.
d = dict(zip (l_words, l_type))

d['Gimsøy']

wt[2220]

len (l_words)

s = set (l_words)

l_words = list (s)

len (l_words)

import pandas as pd
places = pd.Series(l_words)

places.to_pickle(DATA_DIR + "\\stedsnavn.pkl")

# Save point

s = pd.read_pickle(DATA_DIR + "\\stedsnavn.pkl")

"Gråten" in s.values





